"""Domain logic: plate normalization, aggregation, dedupe, trip matching."""

from packages.shared.domain.aggregation import (
    FinalPlateDecision,
    OCRFrameCandidate,
    resolve_plate_candidates,
)
from packages.shared.domain.dedupe import DirectionDeduper
from packages.shared.domain.plate import (
    is_indian_plate_format,
    is_valid_plate_format,
    normalize_plate_text,
)
from packages.shared.domain.trip_matcher import (
    GateEventForMatching,
    TripMatcher,
    TripRecord,
    TripUpdateResult,
)

__all__ = [
    "DirectionDeduper",
    "FinalPlateDecision",
    "GateEventForMatching",
    "OCRFrameCandidate",
    "TripMatcher",
    "TripRecord",
    "TripUpdateResult",
    "is_indian_plate_format",
    "is_valid_plate_format",
    "normalize_plate_text",
    "resolve_plate_candidates",
]
