"""Shared application settings loaded from environment / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration shared across every service in the monorepo."""

    # ── Application ──────────────────────────────────────────────
    APP_NAME: str = "SchoolVF"
    ENV: str = "development"
    DEBUG: bool = True

    # ── Database (PostgreSQL + asyncpg) ──────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/schoolvf"

    # ── Redis ────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Storage ──────────────────────────────────────────────────
    STORAGE_ROOT: str = "./storage"
    SNAPSHOT_DIR: str = "./storage/snapshots"

    # ── Vision / Detection ───────────────────────────────────────
    DEFAULT_SAMPLING_FPS: float = 2.0
    DETECTOR_CONFIDENCE_THRESHOLD: float = 0.6
    OCR_CONFIDENCE_THRESHOLD: float = 0.5

    # ── Trip logic ───────────────────────────────────────────────
    TRIP_OVERDUE_MINUTES: int = 120
    DEDUPE_WINDOW_SECONDS: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
