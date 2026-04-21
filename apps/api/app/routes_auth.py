"""Public auth routes (login)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from jose import jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppUser
from apps.api.app.db.session import get_db
from apps.api.app.deps import get_current_username
from apps.api.app.identity import normalize_username
from apps.api.app.password_util import hash_password
from apps.api.app.settings import get_api_settings

router = APIRouter()


class LoginBody(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserMe(BaseModel):
    username: str


@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginBody, db: Session = Depends(get_db)) -> TokenResponse:
    settings = get_api_settings()
    uname = normalize_username(body.username)
    if not uname:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    authenticated = False
    user = (
        db.query(AppUser)
        .filter(AppUser.username == uname, AppUser.is_active.is_(True))
        .first()
    )
    if user and user.password_hash == hash_password(body.password):
        authenticated = True
        user.last_login = datetime.now(timezone.utc)
        db.commit()

    if not authenticated:
        if uname == normalize_username(settings.AUTH_USERNAME) and body.password == settings.AUTH_PASSWORD:
            authenticated = True

    if not authenticated:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    expire_ts = int(expire.timestamp())
    token = jwt.encode(
        {"sub": uname, "exp": expire_ts},
        settings.JWT_SECRET,
        algorithm="HS256",
    )
    return TokenResponse(access_token=token)


@router.get("/auth/me", response_model=UserMe)
def me(username: str = Depends(get_current_username)) -> UserMe:
    return UserMe(username=username)
