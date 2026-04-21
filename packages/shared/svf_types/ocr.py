"""OCR-related shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRResult:
    """Result from OCR processing of a plate crop."""

    text_raw: str
    confidence: float
    bbox_in_crop: Optional[tuple[int, int, int, int]] = None

    @property
    def text_cleaned(self) -> str:
        """Return text with common noise characters removed."""
        cleaned = self.text_raw.strip().upper()
        # Remove common OCR noise characters
        for char in [" ", "-", ".", ",", "|", "\\", "/"]:
            cleaned = cleaned.replace(char, "")
        return cleaned
