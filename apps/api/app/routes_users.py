"""User management (PostgreSQL)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppUser
from apps.api.app.db.session import get_db
from apps.api.app.deps import get_current_username
from apps.api.app.identity import normalize_username
from apps.api.app.password_util import hash_password

router = APIRouter()


class UserOut(BaseModel):
    id: str
    username: str
    display_name: str
    role: str
    is_active: bool
    created_at: str
    last_login: str | None


class CreateUserIn(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    role: str = "operator"

    @field_validator("username")
    @classmethod
    def _username_norm(cls, v: str) -> str:
        s = normalize_username(v)
        if not s:
            raise ValueError("username required")
        return s

    @field_validator("display_name")
    @classmethod
    def _display_strip(cls, v: str) -> str:
        return v.strip()


class UpdateUserIn(BaseModel):
    display_name: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = None


def _to_out(u: AppUser) -> UserOut:
    return UserOut(
        id=u.id,
        username=u.username,
        display_name=u.display_name,
        role=u.role,
        is_active=u.is_active,
        created_at=u.created_at.isoformat(),
        last_login=u.last_login.isoformat() if u.last_login else None,
    )


def _require_admin(db: Session, username: str) -> AppUser:
    uname = normalize_username(username)
    u = db.query(AppUser).filter(AppUser.username == uname, AppUser.is_active.is_(True)).first()
    if not u or u.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return u


@router.get("/users/me/profile", response_model=UserOut)
def my_profile(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
) -> UserOut:
    uname = normalize_username(username)
    u = db.query(AppUser).filter(AppUser.username == uname, AppUser.is_active.is_(True)).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_out(u)


@router.get("/users", response_model=list[UserOut])
def list_users(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
) -> list[UserOut]:
    _require_admin(db, username)
    rows = db.query(AppUser).order_by(AppUser.created_at.asc()).all()
    return [_to_out(u) for u in rows]


@router.post("/users", response_model=UserOut)
def create_user(
    body: CreateUserIn,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
) -> UserOut:
    _require_admin(db, username)
    if db.query(AppUser).filter(AppUser.username == body.username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    now = datetime.now(timezone.utc)
    u = AppUser(
        id=uuid4().hex[:12],
        username=body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        role=body.role,
        is_active=True,
        created_at=now,
        last_login=None,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return _to_out(u)


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: str,
    body: UpdateUserIn,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
) -> UserOut:
    _require_admin(db, username)
    u = db.query(AppUser).filter(AppUser.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if body.display_name is not None:
        u.display_name = body.display_name.strip()
    if body.role is not None:
        u.role = body.role
    if body.is_active is not None:
        u.is_active = body.is_active
    if body.password is not None:
        u.password_hash = hash_password(body.password)
    db.commit()
    db.refresh(u)
    return _to_out(u)


@router.delete("/users/{user_id}", response_model=UserOut)
def delete_user(
    user_id: str,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
) -> UserOut:
    caller = _require_admin(db, username)
    if caller.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    u = db.query(AppUser).filter(AppUser.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    u.is_active = False
    db.commit()
    db.refresh(u)
    return _to_out(u)


# Back-compat for routes_auth (hash_password)
def _hash_password(password: str) -> str:
    return hash_password(password)
