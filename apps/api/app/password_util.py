"""Password hashing (shared by auth and user admin)."""

from __future__ import annotations

import hashlib


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
