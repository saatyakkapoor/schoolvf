"""Vehicle registration and route management (PostgreSQL)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppVehicle
from apps.api.app.db.session import get_db
from apps.api.app.deps import get_current_username

router = APIRouter(dependencies=[Depends(get_current_username)])


class VehicleOut(BaseModel):
    id: str
    plate_number: str
    vehicle_type: str
    route_number: str
    route_name: str
    driver_name: str
    driver_phone: str
    capacity: int
    is_active: bool
    created_at: str
    updated_at: str


class CreateVehicleIn(BaseModel):
    plate_number: str
    vehicle_type: str = "bus"
    route_number: str
    route_name: str = ""
    driver_name: str = ""
    driver_phone: str = ""
    capacity: int = 40


class UpdateVehicleIn(BaseModel):
    plate_number: str | None = None
    vehicle_type: str | None = None
    route_number: str | None = None
    route_name: str | None = None
    driver_name: str | None = None
    driver_phone: str | None = None
    capacity: int | None = None
    is_active: bool | None = None


def _to_out(v: AppVehicle) -> VehicleOut:
    return VehicleOut(
        id=v.id,
        plate_number=v.plate_number,
        vehicle_type=v.vehicle_type,
        route_number=v.route_number,
        route_name=v.route_name,
        driver_name=v.driver_name,
        driver_phone=v.driver_phone,
        capacity=v.capacity,
        is_active=v.is_active,
        created_at=v.created_at.isoformat(),
        updated_at=v.updated_at.isoformat(),
    )


def count_active_vehicles(db: Session) -> int:
    return db.query(AppVehicle).filter(AppVehicle.is_active.is_(True)).count()


@router.get("/vehicles/routes")
def list_routes(db: Session = Depends(get_db)) -> list[dict[str, str | int]]:
    route_map: dict[str, dict[str, str | int]] = {}
    for v in db.query(AppVehicle).filter(AppVehicle.is_active.is_(True)).all():
        if v.route_number not in route_map:
            route_map[v.route_number] = {
                "route_number": v.route_number,
                "route_name": v.route_name,
                "vehicle_count": 0,
            }
        route_map[v.route_number]["vehicle_count"] = int(route_map[v.route_number]["vehicle_count"]) + 1
    return list(route_map.values())


@router.get("/vehicles/by-plate/{plate_number}", response_model=VehicleOut)
def get_vehicle_by_plate(plate_number: str, db: Session = Depends(get_db)) -> VehicleOut:
    normalized = plate_number.upper().replace(" ", "")
    v = db.query(AppVehicle).filter(AppVehicle.plate_number == normalized).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return _to_out(v)


@router.get("/vehicles", response_model=list[VehicleOut])
def list_vehicles(
    active_only: bool = Query(False),
    route_number: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[VehicleOut]:
    q = db.query(AppVehicle)
    if active_only:
        q = q.filter(AppVehicle.is_active.is_(True))
    if route_number:
        q = q.filter(AppVehicle.route_number == route_number)
    return [_to_out(v) for v in q.order_by(AppVehicle.route_number.asc()).all()]


@router.get("/vehicles/{vehicle_id}", response_model=VehicleOut)
def get_vehicle(vehicle_id: str, db: Session = Depends(get_db)) -> VehicleOut:
    v = db.query(AppVehicle).filter(AppVehicle.id == vehicle_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return _to_out(v)


@router.post("/vehicles", response_model=VehicleOut)
def create_vehicle(body: CreateVehicleIn, db: Session = Depends(get_db)) -> VehicleOut:
    now = datetime.now(timezone.utc)
    v = AppVehicle(
        id=uuid4().hex[:12],
        plate_number=body.plate_number.upper().replace(" ", ""),
        vehicle_type=body.vehicle_type,
        route_number=body.route_number,
        route_name=body.route_name,
        driver_name=body.driver_name,
        driver_phone=body.driver_phone,
        capacity=body.capacity,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return _to_out(v)


@router.patch("/vehicles/{vehicle_id}", response_model=VehicleOut)
def update_vehicle(vehicle_id: str, body: UpdateVehicleIn, db: Session = Depends(get_db)) -> VehicleOut:
    v = db.query(AppVehicle).filter(AppVehicle.id == vehicle_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    if body.plate_number is not None:
        v.plate_number = body.plate_number.upper().replace(" ", "")
    if body.vehicle_type is not None:
        v.vehicle_type = body.vehicle_type
    if body.route_number is not None:
        v.route_number = body.route_number
    if body.route_name is not None:
        v.route_name = body.route_name
    if body.driver_name is not None:
        v.driver_name = body.driver_name
    if body.driver_phone is not None:
        v.driver_phone = body.driver_phone
    if body.capacity is not None:
        v.capacity = body.capacity
    if body.is_active is not None:
        v.is_active = body.is_active
    v.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(v)
    return _to_out(v)


@router.delete("/vehicles/{vehicle_id}", response_model=VehicleOut)
def delete_vehicle(vehicle_id: str, db: Session = Depends(get_db)) -> VehicleOut:
    v = db.query(AppVehicle).filter(AppVehicle.id == vehicle_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    v.is_active = False
    v.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(v)
    return _to_out(v)
