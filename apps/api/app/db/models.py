"""ORM models for app_* tables."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    display_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AppVehicle(Base):
    __tablename__ = "app_vehicles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    plate_number: Mapped[str] = mapped_column(String(32))
    vehicle_type: Mapped[str] = mapped_column(String(32))
    route_number: Mapped[str] = mapped_column(String(64))
    route_name: Mapped[str] = mapped_column(Text, default="")
    driver_name: Mapped[str] = mapped_column(String(255), default="")
    driver_phone: Mapped[str] = mapped_column(String(64), default="")
    capacity: Mapped[int] = mapped_column(Integer, default=40)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AppGateEvent(Base):
    __tablename__ = "app_gate_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[str] = mapped_column(String(64))
    camera_name: Mapped[str] = mapped_column(String(255), default="")
    gate_type: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(16))
    plate_number: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    snapshot_base64: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class AppCamera(Base):
    __tablename__ = "app_cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    gate_type: Mapped[str] = mapped_column(String(16))
    stream_url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="offline")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AppBusSwap(Base):
    """Records when a bus mismatch is acknowledged and classified by staff."""

    __tablename__ = "app_bus_swaps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(32))        # gate event that triggered the mismatch
    plate_number: Mapped[str] = mapped_column(String(32))
    registered_route: Mapped[str] = mapped_column(String(32))
    detected_route: Mapped[str] = mapped_column(String(32))
    swap_type: Mapped[str] = mapped_column(String(32))       # temporary | permanent | glitch | other
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AppTrip(Base):
    __tablename__ = "app_trips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    plate_number: Mapped[str] = mapped_column(String(32))
    exit_event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    entry_event_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    anomaly_code: Mapped[str] = mapped_column(String(64), default="none")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
