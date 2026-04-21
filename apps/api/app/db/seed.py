"""Create ORM tables if missing and seed default admin + demo vehicles."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from apps.api.app.db.models import AppUser, AppVehicle, Base
from apps.api.app.db.session import SessionLocal, engine
from apps.api.app.password_util import hash_password
from apps.api.app.settings import get_api_settings

log = logging.getLogger("schoolvf.db")


def init_db_tables() -> None:
    # Import side effect: register every DeclarativeBase subclass (app_cameras, app_trips, …)
    # so create_all actually creates all tables — not only models imported elsewhere.
    import apps.api.app.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def seed_if_empty() -> None:
    init_db_tables()
    settings = get_api_settings()
    db = SessionLocal()
    try:
        if db.query(AppUser).count() > 0:
            return
        now = datetime.now(timezone.utc)
        admin = AppUser(
            id=uuid4().hex[:12],
            username=settings.AUTH_USERNAME.strip().lower(),
            password_hash=hash_password(settings.AUTH_PASSWORD),
            display_name="Administrator",
            role="admin",
            is_active=True,
            created_at=now,
            last_login=None,
        )
        db.add(admin)
        for v in _demo_vehicles(now):
            db.add(v)
        db.commit()
        log.info("Seeded default admin and demo vehicles")
    except Exception as e:
        log.exception("DB seed failed: %s", e)
        db.rollback()
    finally:
        db.close()


def _demo_vehicles(now: datetime) -> list[AppVehicle]:
    return [
        AppVehicle(
            id=uuid4().hex[:12],
            plate_number="KA01AB1234",
            vehicle_type="bus",
            route_number="R-01",
            route_name="Sector 56 - DLF Phase 4 → School",
            driver_name="Ramesh Kumar",
            driver_phone="",
            capacity=40,
            is_active=True,
            created_at=now,
            updated_at=now,
        ),
        AppVehicle(
            id=uuid4().hex[:12],
            plate_number="DL03CB9012",
            vehicle_type="bus",
            route_number="R-02",
            route_name="Sector 29 - Huda City Centre → School",
            driver_name="Suresh Singh",
            driver_phone="",
            capacity=45,
            is_active=True,
            created_at=now,
            updated_at=now,
        ),
        AppVehicle(
            id=uuid4().hex[:12],
            plate_number="HR26DK7890",
            vehicle_type="van",
            route_number="R-03",
            route_name="Golf Course Road → School",
            driver_name="Mohan Lal",
            driver_phone="",
            capacity=15,
            is_active=True,
            created_at=now,
            updated_at=now,
        ),
    ]
