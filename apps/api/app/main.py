"""FastAPI entrypoint for SchoolVF."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.app.db.cameras import ensure_default_cameras
from apps.api.app.db.seed import seed_if_empty
from apps.api.app.db.session import SessionLocal
from apps.api.app.routes import router as api_router
from apps.api.app.routes_auth import router as auth_router
from apps.api.app.routes_cameras import router as cameras_router
from apps.api.app.routes_internal import router as internal_router
from apps.api.app.routes_live import router as live_router
from apps.api.app.routes_users import router as users_router
from apps.api.app.routes_vehicles import router as vehicles_router

log = logging.getLogger("schoolvf.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_if_empty()
    db = SessionLocal()
    try:
        ensure_default_cameras(db)
    except Exception:
        log.exception("ensure_default_cameras failed")
        db.rollback()
    finally:
        db.close()
    yield


app = FastAPI(title="SchoolVF API", version="0.1.0", lifespan=lifespan)

# allow_credentials=False so allow_origins=["*"] is valid in browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(internal_router, prefix="/api")
app.include_router(live_router, prefix="/api")
app.include_router(cameras_router, prefix="/api")
app.include_router(vehicles_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(api_router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
def health_prefixed() -> dict[str, str]:
    return {"status": "ok"}
