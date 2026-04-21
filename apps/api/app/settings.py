"""API settings (env + defaults)."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class ApiSettings(BaseSettings):
    """Auth and JWT — override via environment in production."""

    # PostgreSQL (async URL in worker; API uses sync psycopg2 driver)
    DATABASE_URL: str = "postgresql+asyncpg://schoolvf:schoolvf_secret@localhost:5432/schoolvf"

    JWT_SECRET: str = "schoolvf-dev-change-in-production"
    JWT_EXPIRE_MINUTES: int = 60 * 24
    AUTH_USERNAME: str = "tsrs"
    AUTH_PASSWORD: str = "TSRS@2026"
    # Vision worker -> API ingestion (set same in worker env)
    INTERNAL_INGEST_SECRET: str = "dev-ingest-secret"
    # MJPEG Live Monitor: used when a camera's stored stream_url has no rtsp://user:pass@ (encode @ in password as %40)
    CAMERA_RTSP_URL: str = (
        "rtsp://admin:abcd%401234@192.168.1.12:554/Streaming/Channels/101"
    )
    # Seed for cam-entry-1 when DB is empty (edit in UI or ENTRY_CAMERA_RTSP_DEFAULT env)
    ENTRY_CAMERA_RTSP_DEFAULT: str = "rtsp://example/entry"
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_api_settings() -> ApiSettings:
    return ApiSettings()


def database_url_sync() -> str:
    """SQLAlchemy sync driver URL (asyncpg → psycopg2)."""
    u = get_api_settings().DATABASE_URL
    if "+asyncpg" in u:
        return u.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    if u.startswith("postgresql://"):
        return u.replace("postgresql://", "postgresql+psycopg2://", 1)
    return u
