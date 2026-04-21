"""Compatibility package for local runs.

Maps ``apps.vision_worker`` imports to source files in ``apps/vision-worker``.
Docker already copies that folder to ``/app/apps/vision_worker``.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "vision-worker"
__path__ = [str(_SRC)]
