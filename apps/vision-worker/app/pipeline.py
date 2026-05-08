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

from apps.vision_worker.app.draw_overlay import (
    BoxRecord,
    make_route_box,
    render_overlay,
)


def _annotate_frame(
    frame: "np.ndarray",
    plate_boxes: list,
    placard_bbox: tuple | None,
    route_text: str | None,
) -> "np.ndarray":
    """Compose the per-frame overlay (plate boxes + route placard) and burn
    it onto a copy of `frame`. Used right before snapshot encoding so the
    image the dashboard receives shows what was detected.
    """
    boxes: list = list(plate_boxes or [])
    if placard_bbox is not None:
        # placard_bbox = (x1, y1, x2, y2, conf)
        try:
            x1, y1, x2, y2, pconf = placard_bbox
            boxes.append(make_route_box(
                (x1, y1, x2, y2),
                text=route_text or "placard",
                conf=float(pconf) if pconf else None,
            ))
        except Exception:
            pass
    if not boxes:
        return frame
    return render_overlay(frame, boxes)
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


def _union_vehicle_bbox(
    boxes: list[tuple[int, int, int, int]],
    fh: int,
    fw: int,
) -> tuple[int, int, int, int]:
    """Axis-aligned union of all vehicle boxes plus a small margin."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad = max(8, int(0.02 * max(x2 - x1, y2 - y1)))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(fw, x2 + pad),
        min(fh, y2 + pad),
    )


def _vehicle_boxes_for_route(frame: "np.ndarray") -> list[tuple[int, int, int, int]]:
    """One lightweight vehicle-YOLO pass so route OCR knows where the bus is.

    This duplicates the vehicle detection that also runs inside the plate
    stack, but that path is on another thread — we need bboxes *before*
    submitting route OCR. Cost is one extra YOLO forward per frame; on a
    T1000 it is the right trade for spatially correct placard reading."""
    try:
        from apps.vision_worker.app.plate_detection import get_vehicle_detector

        vehs = get_vehicle_detector().detect(frame)
        return [tuple(int(x) for x in v["bbox"]) for v in vehs[:6]]
    except Exception as exc:
        log.debug("vehicle pre-pass for route failed: %s", exc)
        return []


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


def _read_route_number(
    frame: "np.ndarray",
    vehicle_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[str | None, "tuple | None"]:
    """
    Route reader for yellow AR-XX placards / LED strip — **only inside vehicle
    detections**. Searching the full frame was picking up random yellow signs,
    road paint, and buildings; the user reported bogus routes from "somewhere
    random".

    Parameters
    ----------
    vehicle_bboxes
        Pixel bboxes ``(x1,y1,x2,y2)`` from the same-frame vehicle YOLO pass.
        If empty / None, route OCR is skipped entirely (no bus → no route).

    Returns (route, placard_bbox) — placard_bbox is frame coords for overlay.
    """
    try:
        from apps.vision_worker.app.plate_engine import _get_ocr

        ocr = _get_ocr()
        h, w = frame.shape[:2]

        # User explicit rule: route number must be reported for EVERY bus,
        # even when vehicle YOLO mis-classifies the bus and we have no bbox.
        # When no vehicle box is available we treat the entire frame as one
        # virtual bus, so the yellow-placard search still runs. False
        # positives are mostly inert because (a) the AR-XX regex is strict
        # and (b) the dashboard surfaces the route as a "registry suggests"
        # hint rather than asserting a plate.
        if not vehicle_bboxes:
            log.debug("route OCR: no vehicle boxes — falling back to full-frame placard scan")
            vehicle_bboxes = [(0, 0, w, h)]

        ux1, uy1, ux2, uy2 = _union_vehicle_bbox(vehicle_bboxes, h, w)
        union_area = max(1, (ux2 - ux1) * (uy2 - uy1))

        # Binary mask: yellow placard must lie ON the detected bus(es).
        bus_mask = np.zeros((h, w), dtype=np.uint8)
        for bx1, by1, bx2, by2 in vehicle_bboxes:
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(bus_mask, (bx1, by1), (bx2, by2), 255, -1)
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        bus_mask = cv2.dilate(bus_mask, dil_k)

        best_route: str | None = None
        best_area: float = 0.0
        placard_bbox: tuple | None = None

        # ------------------------------------------------------------------ #
        # Yellow-bus detection: when the bus body itself is yellow (e.g.
        # all Aravali school buses), the HSV mask covers the entire body
        # and the placard contour gets fused with the body or skipped by
        # the size cap. We detect this case by measuring how much of the
        # bus crop is yellow — anything > 35% is "yellow-body" and we
        # disable the bare-number fallback for the yellow-mask pass to
        # prevent stickers / paint markings from masquerading as routes.
        # ------------------------------------------------------------------ #
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask_full = cv2.inRange(hsv, _YELLOW_HSV_LO, _YELLOW_HSV_HI)
        yellow_inside_bus = cv2.bitwise_and(yellow_mask_full, bus_mask)
        bus_area = float(np.count_nonzero(bus_mask)) or 1.0
        yellow_frac = float(np.count_nonzero(yellow_inside_bus)) / bus_area
        is_yellow_body = yellow_frac > 0.35
        # On yellow-body buses the placard is identified by the AR-XX
        # text only, never by being "the yellow rectangle". Bare numbers
        # are too false-prone (every "5", "12", or sticker digit becomes
        # a route).
        allow_bare = not is_yellow_body
        if is_yellow_body:
            log.debug("route OCR: yellow-body bus (yellow_frac=%.2f) — strict AR-XX only", yellow_frac)

        # All passes append into this list and we pick the highest-score
        # winner at the end. Each pass is deliberately CHEAP: small ROI,
        # one OCR call, strict regex via _ocr_scan_for_route.
        candidates: list[tuple[str, float, str]] = []  # (route, score, source)

        def _try_pass(label: str, x1: int, y1: int, x2: int, y2: int,
                      *, upscale: float = 2.5, sharpen: float = 1.45,
                      allow_bare_for_pass: bool = False) -> None:
            x1c = max(0, int(x1)); y1c = max(0, int(y1))
            x2c = min(w, int(x2)); y2c = min(h, int(y2))
            if x2c - x1c < 12 or y2c - y1c < 8:
                return
            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                return
            try:
                if upscale and upscale > 1.0:
                    roi = cv2.resize(
                        roi,
                        (int(roi.shape[1] * upscale), int(roi.shape[0] * upscale)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                if sharpen and sharpen > 0:
                    roi = _sharpen_roi(roi, strength=float(sharpen))
                result, _ = ocr(roi)
            except Exception as exc:
                log.debug("route OCR pass %s failed: %s", label, exc)
                return
            if not result:
                return
            r, a = _ocr_scan_for_route(result, label, allow_bare_number=allow_bare_for_pass)
            if r and a > 0:
                candidates.append((r, a, label))
            # Maintain the placard_bbox using the most-confident pass that hit.
            nonlocal placard_bbox
            if r and a > 0 and (placard_bbox is None or a > placard_bbox[4]):
                try:
                    pconf = max(float(l[2]) for l in result if len(l) >= 3)
                except (ValueError, IndexError):
                    pconf = 0.0
                placard_bbox = (x1c, y1c, x2c, y2c, pconf)

        # ------------------------------------------------------------------ #
        # Pass A: yellow-contour ROI (only when the bus body ISN'T yellow,
        # otherwise the contour is the whole bus and the OCR finds nothing
        # useful). Uses strict AR-XX regex on yellow-body buses.
        # ------------------------------------------------------------------ #
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        yellow_mask = cv2.morphologyEx(yellow_inside_bus, cv2.MORPH_CLOSE, k_close)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, k_open)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and not is_yellow_body:
            best_cnt: tuple[int, int, int, int] | None = None
            best_cnt_score = 0.0
            for cnt in contours:
                cx, cy_box, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                ar = cw / max(ch, 1)
                area_frac = area / union_area
                if area_frac < 0.0008 or area_frac > 0.45:
                    continue
                if ar < 1.2 or ar > 12.0:
                    continue
                centre_y = cy_box + ch / 2.0
                rel_y = (centre_y - uy1) / max(uy2 - uy1, 1)
                if rel_y < 0.04 or rel_y > 0.72:
                    continue
                aspect_score = 1.0 - min(abs(ar - 3.5) / 3.5, 1.0)
                score = area_frac * 8.0 + aspect_score
                if score > best_cnt_score:
                    best_cnt_score = score
                    best_cnt = (cx, cy_box, cw, ch)
            if best_cnt is not None:
                bx, by, bw, bh = best_cnt
                pad_x = max(6, int(bw * 0.10))
                pad_y = max(4, int(bh * 0.20))
                _try_pass(
                    "yellow_mask",
                    bx - pad_x, by - pad_y, bx + bw + pad_x, by + bh + pad_y,
                    upscale=3.0, sharpen=1.8,
                    allow_bare_for_pass=allow_bare,
                )

        # ------------------------------------------------------------------ #
        # Pass B-D: windshield / front-strip bands.
        # The placard sits in one of three places on Aravali school buses:
        #   B. above the windshield (LED display strip)        — top 0-25% of bus
        #   C. inside the windshield (driver-side card)         — top 12-45% of bus
        #   D. front of the body, above the bumper grille       — top 30-60% of bus
        # We OCR all three so a placard mounted in ANY of these works.
        # The strict AR-XX regex (allow_bare_number=False here) prevents
        # body stickers ("ARAVALI SCHOOL", "ON DUTY", "BUS NO. 5") from
        # matching.
        # ------------------------------------------------------------------ #
        vh = uy2 - uy1
        _try_pass("front_top",
                  ux1, uy1, ux2, uy1 + int(vh * 0.25),
                  upscale=2.5, sharpen=1.5, allow_bare_for_pass=False)
        _try_pass("windshield",
                  ux1, uy1 + int(vh * 0.10), ux2, uy1 + int(vh * 0.48),
                  upscale=2.0, sharpen=1.35, allow_bare_for_pass=False)
        _try_pass("front_grille",
                  ux1, uy1 + int(vh * 0.30), ux2, uy1 + int(vh * 0.62),
                  upscale=2.0, sharpen=1.35, allow_bare_for_pass=False)

        # Pick the highest-scoring candidate across all passes.
        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)
            best_route, best_area, src = candidates[0]
            log.info(
                "route OCR winner: %s score=%.0f source=%s (yellow_body=%s, %d passes hit)",
                best_route, best_area, src, is_yellow_body, len(candidates),
            )
        return best_route, placard_bbox
    except Exception as exc:
        log.warning("route OCR exception: %s", exc)
    return None, None

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
    """Persistent OCR worker pool for plate + route futures (prefetch-friendly)."""
    global _ocr_pool
    if _ocr_pool is None:
        try:
            w = max(2, int(get_settings().OCR_POOL_WORKERS))
        except Exception:
            w = 8
        _ocr_pool = ThreadPoolExecutor(max_workers=w, thread_name_prefix="ocr")
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


class _PlateVoter:
    """Multi-frame consensus voter — accumulates OCR reads during a "vehicle visit"
    (one bus passing the gate) and emits a single high-confidence winner.

    Why this exists: a single frame's OCR is noisy. The same plate is seen 5-30
    times during a normal pass; voting across frames trades a tiny latency for
    a huge accuracy win.

    Strategy:
      1. Each call to `add(plate, conf, snapshot, route)` accumulates a candidate.
      2. We score candidates with `sum(conf^2)` so a single 0.95 read beats a
         pile of low-conf garbage. Squaring is the classic trick for this.
      3. Character-position fallback for same-length variants:
            HR55BC2973  (3 votes, conf 0.7 each)
            HR58BC2973  (1 vote, 0.6) — likely OCR misread of "5" → "8"
         char-vote merges them into HR55BC2973 (5 wins position 2 by votes).
      4. `flush()` returns the winner only when the visit is "ripe" — either
         max_age has elapsed or the visit ended (vehicle gone N seconds).
    """

    def __init__(self, *, min_votes: int = 2, max_age_sec: float = 2.5,
                 visit_end_grace_sec: float = 2.0):
        self._min_votes = max(1, int(min_votes))
        self._max_age_sec = float(max_age_sec)
        self._grace_sec = float(visit_end_grace_sec)
        self._reads: list[tuple[float, str, float, str | None, str | None]] = []
        # (timestamp, plate, conf, snapshot_b64, route)
        self._first_t: float | None = None
        self._last_seen_vehicle: float | None = None

    def add(self, plate: str, conf: float, *, snapshot: str | None,
            route: str | None) -> None:
        if not plate or len(plate) < 4:
            return
        now = time.time()
        if self._first_t is None:
            self._first_t = now
        self._reads.append((now, plate, float(conf), snapshot, route))

    def mark_vehicle_seen(self) -> None:
        self._last_seen_vehicle = time.time()

    def has_data(self) -> bool:
        return bool(self._reads)

    def reset(self) -> None:
        self._reads = []
        self._first_t = None
        self._last_seen_vehicle = None

    def _char_vote_merge(self, candidates: list[tuple[str, float]]) -> tuple[str, float] | None:
        """Per-character voting on same-length plates. Returns (text, conf)."""
        if not candidates:
            return None
        # Bucket by length and pick the bucket with the highest summed confidence.
        by_len: dict[int, list[tuple[str, float]]] = {}
        for txt, c in candidates:
            by_len.setdefault(len(txt), []).append((txt, c))
        best_len = max(by_len, key=lambda L: sum(c for _, c in by_len[L]))
        bucket = by_len[best_len]
        merged = []
        total_conf = 0.0
        for i in range(best_len):
            char_scores: dict[str, float] = {}
            for txt, c in bucket:
                ch = txt[i]
                char_scores[ch] = char_scores.get(ch, 0.0) + (c * c)
            best_ch = max(char_scores, key=lambda k: char_scores[k])
            merged.append(best_ch)
            total_conf += char_scores[best_ch]
        if not merged:
            return None
        return ("".join(merged), min(0.99, total_conf / max(1, best_len)))

    def _ripe(self, force: bool = False) -> bool:
        if force:
            return True
        if not self._reads or self._first_t is None:
            return False
        elapsed = time.time() - self._first_t
        # End-of-visit: vehicle disappeared and grace passed.
        if (
            self._last_seen_vehicle is not None
            and (time.time() - self._last_seen_vehicle) >= self._grace_sec
        ):
            return True
        # Bounded latency: even if vehicle is still in frame, don't wait forever.
        if elapsed >= self._max_age_sec and len(self._reads) >= self._min_votes:
            return True
        return False

    def flush(self, *, force: bool = False) -> tuple[str, float, str | None, str | None] | None:
        """Return winning (plate, conf, snapshot, route) and clear state. None if not ripe."""
        if not self._ripe(force=force):
            return None
        # Score candidates: sum of conf^2 per exact text.
        scores: dict[str, float] = {}
        snaps: dict[str, str] = {}
        routes: dict[str, str] = {}
        candidates: list[tuple[str, float]] = []
        for ts, plate, conf, snap, route in self._reads:
            scores[plate] = scores.get(plate, 0.0) + (conf * conf)
            candidates.append((plate, conf))
            # Keep the snapshot/route from the highest-conf read of this text.
            if snap and (plate not in snaps or conf > scores.get(plate + "::conf", 0.0)):
                snaps[plate] = snap
                scores[plate + "::conf"] = conf
            if route and plate not in routes:
                routes[plate] = route

        if not scores:
            self.reset()
            return None
        # Rank exact-text candidates first. Char-merge as a refiner that can
        # *override* if it produces a same-length winner with higher score.
        exact = [(t, s) for t, s in scores.items() if "::conf" not in t]
        exact.sort(key=lambda kv: kv[1], reverse=True)
        winner_text, winner_score = exact[0]
        winner_avg = winner_score ** 0.5  # rough mean conf
        char_merged = self._char_vote_merge(candidates)
        if char_merged:
            cm_text, cm_conf = char_merged
            cm_score = scores.get(cm_text, 0.0)
            if cm_score >= winner_score:
                winner_text, winner_score, winner_avg = cm_text, cm_score, cm_conf

        # Best snapshot for the winner — fall back to any read's snapshot.
        snap = snaps.get(winner_text)
        if snap is None:
            snap = next((r[3] for r in self._reads if r[3]), None)
        route = routes.get(winner_text) or next((r[4] for r in self._reads if r[4]), None)
        out = (winner_text, min(0.99, max(0.0, winner_avg)), snap, route)

        self.reset()
        return out


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


# ─────────────────────────────────────────────────────────────────────────────
# Background plate enhancer
# ─────────────────────────────────────────────────────────────────────────────
# When the live OCR loop posts a route-only detection (we saw the bus + the
# placard but no plate), we save the snapshot and kick off a deep retry on a
# background thread pool. The retry runs *heavier* OCR variants — CLAHE,
# upscaled to 2.5x and 3.5x, sharpened — that we don't run live because
# they're too slow per frame. If the deep pass finds a plate that matches
# the registry's plate-for-this-route (≥4 chars positional or edit-distance ≤ 3),
# we POST a corrected detection. The user sees the original route-only row
# get joined by a confident plate row a second or two later.
_DEEP_OCR_POOL: ThreadPoolExecutor | None = None
_DEEP_OCR_LOCK = threading.Lock()
# Per-(camera, route) suppression so we don't enhance the same bus 20 times.
_DEEP_OCR_RECENT: dict[str, float] = {}
_DEEP_OCR_WINDOW_SEC = 8.0


def _get_deep_ocr_pool() -> ThreadPoolExecutor:
    global _DEEP_OCR_POOL
    if _DEEP_OCR_POOL is None:
        with _DEEP_OCR_LOCK:
            if _DEEP_OCR_POOL is None:
                _DEEP_OCR_POOL = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="deep-ocr",
                )
    return _DEEP_OCR_POOL


def _deep_ocr_dedupe_allow(camera_id: str, route: str | None) -> bool:
    key = f"{camera_id}::{route or '?'}"
    now = time.time()
    prev = _DEEP_OCR_RECENT.get(key)
    if prev is not None and now - prev < _DEEP_OCR_WINDOW_SEC:
        return False
    _DEEP_OCR_RECENT[key] = now
    return True


def _plates_close_enough(a: str, b: str) -> bool:
    """Same rule as the API's snap-to-registry: ≥4 char position match OR edit dist ≤ 3."""
    if not a or not b:
        return False
    A, B = a.upper(), b.upper()
    matches = sum(1 for i in range(min(len(A), len(B))) if A[i] == B[i])
    if matches >= 4:
        return True
    # Cheap inline edit distance.
    if A == B:
        return True
    if abs(len(A) - len(B)) > 3:
        return False
    prev = list(range(len(B) + 1))
    for i, ca in enumerate(A, 1):
        curr = [i]
        for j, cb in enumerate(B, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1] <= 3


def _enhance_variants(frame: "np.ndarray") -> list["np.ndarray"]:
    """Cheap-to-medium image enhancements for the deep retry. Each variant
    is fed to `read_plates_from_frame` independently.

    Order matters: variants with the highest expected lift come first so
    we can short-circuit on a registry match.
    """
    out: list[np.ndarray] = [frame]
    h, w = frame.shape[:2]
    try:
        # Variant 1: CLAHE on the L channel — boosts plate-on-bus-body contrast
        # without saturating colours.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_, a_, b_ = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_)
        out.append(cv2.cvtColor(cv2.merge((l_eq, a_, b_)), cv2.COLOR_LAB2BGR))
    except Exception:
        pass
    try:
        # Variant 2: 1.5× upscale + sharpen — lets EasyOCR's small-text
        # branch recover characters smudged at native resolution.
        big = cv2.resize(frame, (int(w * 1.5), int(h * 1.5)),
                         interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        out.append(cv2.filter2D(big, -1, kernel))
    except Exception:
        pass
    try:
        # Variant 3: bilateral denoise — kills the JPEG mosquito noise
        # around the plate edges that confuses character segmentation.
        out.append(cv2.bilateralFilter(frame, d=5, sigmaColor=60, sigmaSpace=60))
    except Exception:
        pass
    return out


def _deep_ocr_worker(
    api_base: str,
    secret: str,
    snapshot_jpeg_bytes: bytes,
    camera_id: str,
    camera_name: str,
    route: str | None,
    expected_plate: str | None,
    min_confidence: float,
    strict_indian: bool,
    detection_mode: str,
    vision_stack: str,
) -> None:
    """Run heavier OCR variants on a saved snapshot. POSTs a corrected
    detection when a plate emerges that matches the registry's expectation."""
    try:
        arr = np.frombuffer(snapshot_jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("deep-ocr: failed to decode snapshot for camera %s", camera_id)
            return
    except Exception as exc:
        log.warning("deep-ocr: snapshot decode error: %s", exc)
        return

    candidates: list[tuple[str, float]] = []
    matched: tuple[str, float] | None = None
    for i, variant in enumerate(_enhance_variants(frame)):
        try:
            plates, _boxes = read_plates_from_frame(
                variant,
                min_confidence=min_confidence,
                strict_indian=strict_indian,
                detection_mode=detection_mode,
                vision_stack=vision_stack,
            )
        except Exception as exc:
            log.debug("deep-ocr variant %d failed: %s", i, exc)
            continue
        for plate, conf in plates or []:
            candidates.append((plate, conf))
            if expected_plate and _plates_close_enough(plate, expected_plate):
                matched = (plate, conf)
                break
        if matched:
            break

    if not candidates:
        log.info("deep-ocr [%s route=%s]: no plates from %d variants",
                 camera_id, route or "?", len(_enhance_variants(frame)))
        return

    # Pick the registry-matching candidate when available, else the highest-conf.
    if matched is None and candidates:
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        matched = candidates[0]

    plate, conf = matched
    snap_b64 = None
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            import base64
            snap_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass

    body: dict[str, Any] = {
        "plate_text": plate,
        "confidence": float(conf),
        "camera_id": camera_id,
        "camera_name": camera_name + " · enhanced",
        "snapshot_base64": snap_b64,
    }
    if route:
        body["detected_route"] = route
    try:
        client = _get_http_client()
        r = client.post(
            f"{api_base}/api/live/detections",
            json=body,
            headers={"X-Internal-Token": secret},
        )
        r.raise_for_status()
        log.info(
            "deep-ocr [%s] POSTED plate=%s conf=%.2f route=%s registry_match=%s",
            camera_id, plate, conf, route or "?", expected_plate or "?",
        )
    except Exception as exc:
        log.warning("deep-ocr POST failed: %s", exc)


def _enqueue_deep_ocr(
    s: VisionSettings,
    *,
    snapshot_b64: str | None,
    camera_id: str,
    camera_name: str,
    route: str | None,
    expected_plate: str | None,
) -> None:
    """Kick off a heavy retry on the snapshot. Cheap to call (queue submit)."""
    if not snapshot_b64:
        return
    if not _deep_ocr_dedupe_allow(camera_id, route):
        return
    try:
        import base64
        snap_bytes = base64.b64decode(snapshot_b64)
    except Exception:
        return
    pool = _get_deep_ocr_pool()
    pool.submit(
        _deep_ocr_worker,
        s.API_BASE_URL.rstrip("/"),
        s.INTERNAL_INGEST_SECRET,
        snap_bytes,
        camera_id,
        camera_name,
        route,
        expected_plate,
        s.MIN_CONFIDENCE,
        s.PLATE_FILTER.strip().lower() == "indian",
        s.PLATE_DETECTION_MODE.strip().lower(),
        s.VISION_STACK,
    )


def _registry_plate_for_route(s: VisionSettings, route: str) -> str | None:
    """Best-effort registry lookup: GET the API for the registered plate of a route.
    Used to give the deep-OCR worker a 'target' to verify against."""
    if not route:
        return None
    try:
        client = _get_http_client()
        r = client.get(
            f"{s.API_BASE_URL.rstrip('/')}/api/internal/vehicles-by-route",
            params={"route": route},
            headers={"X-Internal-Token": s.INTERNAL_INGEST_SECRET},
            timeout=2.0,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return (data.get("plate_number") or "").strip().upper() or None
    except Exception as exc:
        log.debug("registry lookup failed (non-fatal): %s", exc)
    return None


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
            # Multi-frame plate voter: collects OCR reads across frames and emits
            # one consensus winner per visit. Default settings tuned for ~30 fps
            # gate cameras: 2 minimum votes, 2.5s max latency.
            plate_voter = _PlateVoter(min_votes=2, max_age_sec=2.5,
                                      visit_end_grace_sec=2.0)

            # ── Prefetch / double-buffered OCR pipeline ──────────────────────
            # While we post / jpeg-encode the *current* frame's results, the
            # *next* frame's plate-OCR + route-OCR is already running on the
            # GPU. This is what keeps the T1000 saturated; without it the GPU
            # sits idle for ~150 ms per cycle (HTTP post + snapshot encode).
            # `pending` holds the OCR work submitted for an upcoming frame.
            pending: dict | None = None
            strict_indian = s.PLATE_FILTER.strip().lower() == "indian"
            _pool = _get_ocr_pool()
            # Gate timing: first vehicle sighting starts a timer; OCR waits until the bus advances.
            vehicle_episode_t0: float | None = None
            vehicle_settled: bool = False

            def _submit_frame(
                _f,
                _captured_at: float,
                vboxes_pre: list[tuple[int, int, int, int]] | None = None,
            ) -> dict:
                """Kick off plate + route OCR for one frame, return the futures."""
                _f_copy = _f.copy()
                t0 = time.time()
                # Vehicle boxes synchronously so route placard OCR never scans
                # the full frame (random yellow signs / road markings).
                vboxes = (
                    list(vboxes_pre)
                    if vboxes_pre is not None
                    else _vehicle_boxes_for_route(_f)
                )
                return {
                    "frame": _f,
                    "frame_copy": _f_copy,
                    "captured_at": _captured_at,
                    "submitted_at": t0,
                    "vehicle_boxes": vboxes,
                    "plate_fut": _pool.submit(
                        read_plates_from_frame,
                        _f,
                        min_confidence=s.MIN_CONFIDENCE,
                        strict_indian=strict_indian,
                        detection_mode=s.PLATE_DETECTION_MODE.strip().lower(),
                        vision_stack=s.VISION_STACK,
                    ),
                    "route_fut": _pool.submit(_read_route_number, _f_copy, vboxes),
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
                    settle_sec = float(s.GATE_VEHICLE_SETTLE_SEC)
                    min_area_frac = float(s.GATE_VEHICLE_MIN_AREA_FRAC)
                    max_wait_sec = float(s.GATE_MAX_WAIT_SEC)
                    vboxes_pre: list[tuple[int, int, int, int]] | None = None
                    if settle_sec > 0 or min_area_frac > 0:
                        vboxes_pre = _vehicle_boxes_for_route(frame)
                        if not vboxes_pre:
                            vehicle_episode_t0 = None
                            vehicle_settled = False
                            # Visit ended — emit consensus winner if enough reads were collected.
                            voted_end = plate_voter.flush(force=True)
                            if voted_end is not None:
                                v_plate, v_conf, v_snap, v_route = voted_end
                                if _dedupe_allow(v_plate, last, s.DEDUPE_SECONDS, v_route) \
                                        and _camera_cooldown_allow(v_plate, cid, recent_cam, s.CAMERA_COOLDOWN_SEC):
                                    _post_detection(
                                        s, v_plate, v_conf, v_snap or None,
                                        camera_id=cid, camera_name=cname,
                                        detected_route=v_route,
                                    )
                                    _camera_cooldown_mark(v_plate, cid, recent_cam)
                                    log.info("visit-end vote: %s conf=%.2f", v_plate, v_conf)
                            time.sleep(0.02)
                            continue
                        if vehicle_episode_t0 is None:
                            vehicle_episode_t0 = time.monotonic()
                            vehicle_settled = False
                            log.info(
                                "camera %s gate: vehicle seen — settle=%.2fs min_area=%.2f",
                                cid, settle_sec, min_area_frac,
                            )
                        if not vehicle_settled:
                            elapsed = time.monotonic() - vehicle_episode_t0
                            # Distance gate: largest vehicle bbox area / frame area.
                            fh_, fw_ = frame.shape[:2]
                            biggest = 0.0
                            if vboxes_pre:
                                for vb in vboxes_pre:
                                    vx1, vy1, vx2, vy2 = vb
                                    a = max(0, vx2 - vx1) * max(0, vy2 - vy1)
                                    if a > biggest:
                                        biggest = a
                            area_frac = biggest / float(fh_ * fw_) if fw_ and fh_ else 0.0
                            time_ok = elapsed >= settle_sec
                            area_ok = (min_area_frac <= 0) or (area_frac >= min_area_frac)
                            timeout = elapsed >= max_wait_sec
                            if not ((time_ok and area_ok) or timeout):
                                time.sleep(0.04)
                                continue
                            vehicle_settled = True
                            log.info(
                                "camera %s gate ready: elapsed=%.2fs area_frac=%.3f%s",
                                cid, elapsed, area_frac,
                                " (timeout)" if timeout and not (time_ok and area_ok) else "",
                            )
                    pending = _submit_frame(frame, captured_at, vboxes_pre=vboxes_pre)

                # Step 2: wait for the in-flight plate OCR to finish.
                current = pending
                pending = None  # we're consuming it now
                frame = current["frame"]
                frame_copy = current["frame_copy"]
                captured_at = current["captured_at"]
                t_ocr = current["submitted_at"]
                vboxes_dbg = current.get("vehicle_boxes") or []
                _plate_fut = current["plate_fut"]
                _route_fut = current["route_fut"]

                plates, plate_boxes = _plate_fut.result()

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
                placard_bbox = None
                try:
                    # Tight timeout: route OCR is already running in parallel.
                    # If it's slower than 1s we skip the route for this frame.
                    route, placard_bbox = _route_fut.result(timeout=1.0)
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

                # Annotate the frame with detection boxes ONCE so every
                # downstream JPEG (debug snapshot, ingest snapshot) shows
                # exactly what the worker found. This is the "proof of work"
                # the user explicitly asked for.
                annotated = _annotate_frame(frame, plate_boxes, placard_bbox, route)

                # Push frame snapshot to debug panel every ~1 s (was every 5
                # frames ≈ every 150 ms — wasteful when nothing's happening).
                # Always push when we have plates or a route hit.
                # Use a 720 px tile so the user can actually READ the plate
                # and verify the overlay boxes; the debug panel shrinks it
                # in CSS but the underlying pixels stay sharp.
                if plates or route or frame_n % 30 == 1:
                    snap_b64 = frame_to_jpeg_b64(annotated, 720)
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
                            "n_plate_boxes": len(plate_boxes or []),
                            "n_vehicles": len(vboxes_dbg),
                            "has_placard_box": placard_bbox is not None,
                        },
                        bypass_interval=bool(plates),
                    )

                snap: str | None = None
                # Confidence floor for *individual* OCR reads going into the voter.
                INGEST_CONF_FLOOR = float(s.INGEST_MIN_CONFIDENCE)
                if vboxes_dbg:
                    plate_voter.mark_vehicle_seen()

                # ── ROUTE FIRST ─────────────────────────────────────────────
                # Post the route the moment we see one, before the plate
                # voter ripens. This guarantees every bus whose placard is
                # visible shows up on the dashboard even if the plate is
                # permanently unreadable from this angle. Repeats are
                # suppressed purely by `_route_only_dedupe_allow` (per-camera
                # +per-route window) so no extra in-visit flag is needed.
                if route:
                    if _route_only_dedupe_allow(route, cid, last, max(s.DEDUPE_SECONDS, 6.0)):
                        if snap is None:
                            snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
                        _post_detection(
                            s, None, 0.0, snap or None,
                            camera_id=cid, camera_name=cname, detected_route=route,
                        )
                        log.info("route post (priority): route=%s camera=%s", route, cid)
                        # Background deep-OCR retry on this snapshot. The
                        # registry plate (if any) is fetched async and used
                        # as a verification target inside the worker.
                        expected = _registry_plate_for_route(s, route)
                        _enqueue_deep_ocr(
                            s,
                            snapshot_b64=snap,
                            camera_id=cid,
                            camera_name=cname,
                            route=route,
                            expected_plate=expected,
                        )

                # ── PLATE VOTING ────────────────────────────────────────────
                for plate, conf in plates:
                    if conf < INGEST_CONF_FLOOR:
                        log.info(
                            "plate suppress: %s conf=%.2f < %.2f (INGEST_MIN_CONFIDENCE)",
                            plate, conf, INGEST_CONF_FLOOR,
                        )
                        continue
                    if snap is None:
                        snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
                    plate_voter.add(plate, conf, snapshot=snap, route=route)

                # Multi-frame voting: emit ONE consensus winner per visit.
                voted = plate_voter.flush(force=False)
                if voted is not None:
                    v_plate, v_conf, v_snap, v_route = voted
                    if _dedupe_allow(v_plate, last, s.DEDUPE_SECONDS, v_route) \
                            and _camera_cooldown_allow(v_plate, cid, recent_cam, s.CAMERA_COOLDOWN_SEC):
                        post_snap = v_snap or snap
                        if post_snap is None:
                            post_snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
                        _post_detection(
                            s, v_plate, v_conf, post_snap or None,
                            camera_id=cid, camera_name=cname, detected_route=v_route,
                        )
                        _camera_cooldown_mark(v_plate, cid, recent_cam)
                        log.info(
                            "voted plate posted: %s conf=%.2f route=%s",
                            v_plate, v_conf, v_route or "?",
                        )
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
            vehicle_episode_t0_wc: float | None = None
            vehicle_settled_wc: bool = False
            plate_voter_wc = _PlateVoter(min_votes=2, max_age_sec=2.5,
                                         visit_end_grace_sec=2.0)

            def _submit_frame_wc(
                _f,
                _captured_at: float,
                vboxes_pre: list[tuple[int, int, int, int]] | None = None,
            ) -> dict:
                t0 = time.time()
                _f_copy = _f.copy()
                vboxes = (
                    list(vboxes_pre)
                    if vboxes_pre is not None
                    else _vehicle_boxes_for_route(_f)
                )
                return {
                    "frame": _f,
                    "frame_copy": _f_copy,
                    "captured_at": _captured_at,
                    "submitted_at": t0,
                    "vehicle_boxes": vboxes,
                    "plate_fut": _pool.submit(
                        read_plates_from_frame,
                        _f,
                        min_confidence=s.MIN_CONFIDENCE,
                        strict_indian=strict_indian,
                        detection_mode=s.PLATE_DETECTION_MODE.strip().lower(),
                        vision_stack=s.VISION_STACK,
                    ),
                    "route_fut": _pool.submit(_read_route_number, _f_copy, vboxes),
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
                    settle_sec = float(s.GATE_VEHICLE_SETTLE_SEC)
                    min_area_frac = float(s.GATE_VEHICLE_MIN_AREA_FRAC)
                    max_wait_sec = float(s.GATE_MAX_WAIT_SEC)
                    vpre: list[tuple[int, int, int, int]] | None = None
                    if settle_sec > 0 or min_area_frac > 0:
                        vpre = _vehicle_boxes_for_route(frame)
                        if not vpre:
                            vehicle_episode_t0_wc = None
                            vehicle_settled_wc = False
                            # Vehicle gone — flush voter so we don't lose a
                            # winner if the bus already produced enough reads.
                            voted_end = plate_voter_wc.flush(force=True)
                            if voted_end is not None:
                                v_plate, v_conf, v_snap, v_route = voted_end
                                if _dedupe_allow(v_plate, last, s.DEDUPE_SECONDS, v_route) \
                                        and _camera_cooldown_allow(v_plate, cid, recent_cam, s.CAMERA_COOLDOWN_SEC):
                                    _post_detection(
                                        s, v_plate, v_conf, v_snap or None,
                                        camera_id=cid, camera_name=cname,
                                        detected_route=v_route,
                                    )
                                    _camera_cooldown_mark(v_plate, cid, recent_cam)
                                    log.info("visit-end vote (webcam): %s conf=%.2f", v_plate, v_conf)
                            time.sleep(0.02)
                            continue
                        if vehicle_episode_t0_wc is None:
                            vehicle_episode_t0_wc = time.monotonic()
                            vehicle_settled_wc = False
                            log.info(
                                "camera %s webcam gate: vehicle seen — settle=%.2fs min_area=%.2f",
                                cid, settle_sec, min_area_frac,
                            )
                        if not vehicle_settled_wc:
                            elapsed = time.monotonic() - vehicle_episode_t0_wc
                            fh_, fw_ = frame.shape[:2]
                            biggest = 0.0
                            for vb in vpre or []:
                                vx1, vy1, vx2, vy2 = vb
                                a = max(0, vx2 - vx1) * max(0, vy2 - vy1)
                                if a > biggest:
                                    biggest = a
                            area_frac = biggest / float(fh_ * fw_) if fw_ and fh_ else 0.0
                            time_ok = elapsed >= settle_sec
                            area_ok = (min_area_frac <= 0) or (area_frac >= min_area_frac)
                            timeout = elapsed >= max_wait_sec
                            if not ((time_ok and area_ok) or timeout):
                                time.sleep(0.04)
                                continue
                            vehicle_settled_wc = True
                            log.info(
                                "camera %s webcam gate ready: elapsed=%.2fs area_frac=%.3f%s",
                                cid, elapsed, area_frac,
                                " (timeout)" if timeout and not (time_ok and area_ok) else "",
                            )
                    pending = _submit_frame_wc(frame, captured_at, vboxes_pre=vpre)

                current = pending
                pending = None
                frame = current["frame"]
                frame_copy = current["frame_copy"]
                captured_at = current["captured_at"]
                t_ocr = current["submitted_at"]
                vboxes_dbg = current.get("vehicle_boxes") or []
                _plate_fut = current["plate_fut"]
                _route_fut = current["route_fut"]

                plates, plate_boxes = _plate_fut.result()

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
                placard_bbox = None
                try:
                    route, placard_bbox = _route_fut.result(timeout=1.0)
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
                annotated = _annotate_frame(frame, plate_boxes, placard_bbox, route)

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
                            "snapshot_b64": frame_to_jpeg_b64(annotated, 720),
                            "n_plate_boxes": len(plate_boxes or []),
                            "n_vehicles": len(vboxes_dbg),
                            "has_placard_box": placard_bbox is not None,
                        },
                    )
                snap: str | None = None
                INGEST_CONF_FLOOR = float(s.INGEST_MIN_CONFIDENCE)
                if vboxes_dbg:
                    plate_voter_wc.mark_vehicle_seen()

                # Route-first: post the moment we see a placard, even if the
                # plate is gibberish. Dedupe window prevents flooding.
                if route:
                    if _route_only_dedupe_allow(route, cid, last, max(s.DEDUPE_SECONDS, 6.0)):
                        if snap is None:
                            snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
                        _post_detection(
                            s, None, 0.0, snap or None,
                            camera_id=cid, camera_name=cname, detected_route=route,
                        )
                        log.info("route post (webcam, priority): route=%s camera=%s", route, cid)
                        expected = _registry_plate_for_route(s, route)
                        _enqueue_deep_ocr(
                            s,
                            snapshot_b64=snap,
                            camera_id=cid,
                            camera_name=cname,
                            route=route,
                            expected_plate=expected,
                        )

                for plate, conf in plates:
                    if conf < INGEST_CONF_FLOOR:
                        log.info(
                            "plate suppress: %s conf=%.2f < %.2f (INGEST_MIN_CONFIDENCE)",
                            plate, conf, INGEST_CONF_FLOOR,
                        )
                        continue
                    if snap is None:
                        snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
                    plate_voter_wc.add(plate, conf, snapshot=snap, route=route)

                voted = plate_voter_wc.flush(force=False)
                if voted is not None:
                    v_plate, v_conf, v_snap, v_route = voted
                    if _dedupe_allow(v_plate, last, s.DEDUPE_SECONDS, v_route) \
                            and _camera_cooldown_allow(v_plate, cid, recent_cam, s.CAMERA_COOLDOWN_SEC):
                        post_snap = v_snap or snap
                        if post_snap is None:
                            post_snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
                        _post_detection(
                            s, v_plate, v_conf, post_snap or None,
                            camera_id=cid, camera_name=cname, detected_route=v_route,
                        )
                        _camera_cooldown_mark(v_plate, cid, recent_cam)
                        log.info("voted plate (webcam): %s conf=%.2f", v_plate, v_conf)

                # Route-only fallback (webcam path) — see RTSP loop for the rationale.
                if not posted_any and route:
                    if _route_only_dedupe_allow(route, cid, last, max(s.DEDUPE_SECONDS, 6.0)):
                        if snap is None:
                            snap = frame_to_jpeg_b64(annotated, s.SNAPSHOT_MAX_WIDTH)
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
