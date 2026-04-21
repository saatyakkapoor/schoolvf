"""Poll API for all active cameras; run one RTSP+OCR loop per camera in parallel (daemon threads)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from apps.vision_worker.app.pipeline import run_rtsp_loop, run_webcam_loop
from apps.vision_worker.app.settings import VisionSettings

log = logging.getLogger("vision.multi_camera")


def _fetch_cameras(api_base: str, secret: str) -> list[dict[str, Any]]:
    base = api_base.rstrip("/")
    # Accept API_BASE_URL=http://api:8000 or http://api:8000/api (avoid double /api/api/).
    if base.lower().endswith("/api"):
        base = base[:-4].rstrip("/")
    # Router is mounted at /api on the FastAPI app.
    url = f"{base}/api/internal/vision-cameras"
    r = httpx.get(
        url,
        headers={"X-Internal-Token": secret},
        timeout=20.0,
    )
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.warning(
            "GET %s failed: HTTP %s — %s (check INTERNAL_INGEST_SECRET matches API)",
            url,
            e.response.status_code,
            (e.response.text or "")[:400],
        )
        raise
    data = r.json()
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _sig(c: dict[str, Any]) -> str:
    return f"{c.get('id')}|{c.get('stream_url')}|{c.get('name')}"


def _env_fallback_camera(s: VisionSettings) -> dict[str, Any] | None:
    """
    When the API returns no cameras (DB empty, ingest secret mismatch, bad URLs),
    still run one pipeline from compose/env so the worker uses CPU and reads plates like before.
    """
    url = (s.CAMERA_RTSP_URL or "").strip()
    url_l = url.lower()
    if not (url_l.startswith("rtsp") or url_l.startswith("webcam:")):
        return None
    cid = (s.CAMERA_ID or "cam-exit-1").strip()
    return {
        "id": cid,
        "name": (s.CAMERA_NAME or "Camera").strip(),
        "stream_url": url,
        "is_active": True,
    }


def run_multi_camera_loop(s: VisionSettings) -> None:
    threads: dict[str, threading.Thread] = {}
    stop_events: dict[str, threading.Event] = {}
    last_sig: dict[str, str] = {}
    poll = max(5.0, float(s.CAMERA_POLL_INTERVAL_SEC))

    while True:
        try:
            raw = _fetch_cameras(s.API_BASE_URL, s.INTERNAL_INGEST_SECRET)
            desired = {str(c["id"]): c for c in raw if c.get("id")}
            if not desired:
                fb = _env_fallback_camera(s)
                if fb:
                    desired = {str(fb["id"]): fb}
                    log.warning(
                        "API returned 0 cameras — using env fallback %s (CAMERA_RTSP_URL/webcam:N). "
                        "Fix DB cameras or INTERNAL_INGEST_SECRET so the API list is used.",
                        fb["id"],
                    )
                else:
                    log.warning(
                        "API returned 0 cameras and CAMERA_RTSP_URL is not a valid rtsp:// URL — "
                        "worker idle. Add cameras in the dashboard (RTSP or webcam:N), or set "
                        "CAMERA_RTSP_URL as env fallback.",
                    )
            if desired:
                log.info(
                    "vision-cameras: %s stream(s): %s",
                    len(desired),
                    ", ".join(desired.keys()),
                )
        except Exception as e:
            log.warning("vision-cameras fetch failed: %s — retry in 10s", e)
            time.sleep(10.0)
            continue

        # Drop cameras removed from API or deactivated
        for cid in list(threads.keys()):
            if cid not in desired:
                log.info("Stopping vision thread for removed/inactive camera %s", cid)
                stop_events[cid].set()
                threads[cid].join(timeout=12.0)
                del threads[cid]
                del stop_events[cid]
                last_sig.pop(cid, None)

        for cid, c in desired.items():
            sig_c = _sig(c)
            if cid in threads and last_sig.get(cid) == sig_c:
                continue
            if cid in threads:
                log.info("Restarting vision thread for camera %s (config changed)", cid)
                stop_events[cid].set()
                threads[cid].join(timeout=12.0)
                del threads[cid]
                del stop_events[cid]

            ev = threading.Event()
            stop_events[cid] = ev
            last_sig[cid] = sig_c
            cm = dict(c)

            def _run(
                settings: VisionSettings,
                cam: dict[str, Any],
                stop: threading.Event,
            ) -> None:
                url = str(cam["stream_url"])
                if url.lower().startswith("webcam:"):
                    try:
                        dev = int(url.split(":", 1)[1])
                    except (ValueError, IndexError):
                        dev = 0
                    run_webcam_loop(
                        settings,
                        camera_id=str(cam["id"]),
                        camera_name=str(cam.get("name") or cam["id"]),
                        device_index=dev,
                        stop_event=stop,
                    )
                else:
                    run_rtsp_loop(
                        settings,
                        camera_id=str(cam["id"]),
                        camera_name=str(cam.get("name") or cam["id"]),
                        rtsp_url=url,
                        stop_event=stop,
                    )

            t = threading.Thread(
                target=_run,
                args=(s, cm, ev),
                name=f"vision-{cid}",
                daemon=True,
            )
            t.start()
            threads[cid] = t
            log.info("Started vision thread for %s (%s)", cid, cm.get("name"))

        time.sleep(poll)
