"""Helpers for app_cameras (DB-backed dashboard cameras)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from apps.api.app.db.models import AppCamera
from apps.api.app.db.seed import init_db_tables
from apps.api.app.settings import get_api_settings

log = logging.getLogger("schoolvf.db")


def ensure_default_cameras(db: Session) -> None:
    """Insert exit + entry cameras if empty (first deploy or new DB)."""
    init_db_tables()
    if db.query(AppCamera).count() > 0:
        return
    settings = get_api_settings()
    now = datetime.now(timezone.utc)
    exit_url = (settings.CAMERA_RTSP_URL or "").strip() or "rtsp://example/exit"
    entry_url = (settings.ENTRY_CAMERA_RTSP_DEFAULT or "").strip() or "rtsp://example/entry"
    # Placeholder entry URL is not a real camera — inactive until user sets RTSP in UI (saves worker CPU).
    entry_placeholder = entry_url.lower() in (
        "rtsp://example/entry",
        "rtsp://example",
    ) or entry_url.lower().startswith("rtsp://example/")
    db.add(
        AppCamera(
            id="cam-exit-1",
            name="Exit gate — Main",
            gate_type="exit",
            stream_url=exit_url,
            status="offline",
            is_active=True,
            last_heartbeat=None,
            created_at=now,
            updated_at=now,
        )
    )
    db.add(
        AppCamera(
            id="cam-entry-1",
            name="Entry gate — Main",
            gate_type="entry",
            stream_url=entry_url,
            status="offline",
            is_active=not entry_placeholder,
            last_heartbeat=None,
            created_at=now,
            updated_at=now,
        )
    )
    db.commit()
    log.info("Seeded default cameras cam-exit-1 and cam-entry-1 (edit RTSP in UI)")
