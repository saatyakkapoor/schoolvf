"""Username normalization for login + storage."""

from __future__ import annotations


def normalize_username(raw: str) -> str:
    return (raw or "").strip().lower()
