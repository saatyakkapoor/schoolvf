"""FastAPI dependencies (auth)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from apps.api.app.settings import get_api_settings

security = HTTPBearer()


def get_current_username(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    settings = get_api_settings()
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
        )
        sub = payload.get("sub")
        if not isinstance(sub, str) or not sub:
            raise HTTPException(status_code=401, detail="Invalid token")
        return sub
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
