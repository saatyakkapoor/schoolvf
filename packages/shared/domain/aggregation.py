"""Multi-frame OCR candidate aggregation into a single final plate decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from packages.shared.domain.plate import is_valid_plate_format, normalize_plate_text


@dataclass(frozen=True)
class OCRFrameCandidate:
    """One frame's OCR + detector scores for aggregation."""

    plate_text_raw: str
    ocr_conf: float
    detector_conf: float

    @property
    def plate_text_norm(self) -> str:
        return normalize_plate_text(self.plate_text_raw)


@dataclass(frozen=True)
class FinalPlateDecision:
    plate_text_norm: str
    confidence_final: float
    needs_review: bool
    winning_vote_count: int
    total_candidates: int


def _score(c: OCRFrameCandidate) -> float:
    return max(0.0, min(1.0, c.ocr_conf)) * max(0.0, min(1.0, c.detector_conf))


def resolve_plate_candidates(
    candidates: Iterable[OCRFrameCandidate],
    *,
    min_confidence: float = 0.5,
    review_confidence: float = 0.65,
) -> FinalPlateDecision:
    """
    Pick final plate using format validation, per-string scoring, and vote counts.

    - Filters to regex-valid normalized strings when possible.
    - Groups candidates by normalized plate; picks group with best aggregate score.
    - Flags low final confidence or inconsistent reads for manual review.
    """
    cands = list(candidates)
    if not cands:
        return FinalPlateDecision(
            plate_text_norm="",
            confidence_final=0.0,
            needs_review=True,
            winning_vote_count=0,
            total_candidates=0,
        )

    valid = [c for c in cands if is_valid_plate_format(c.plate_text_norm)]
    pool = valid if valid else cands

    by_norm: dict[str, list[OCRFrameCandidate]] = {}
    for c in pool:
        by_norm.setdefault(c.plate_text_norm, []).append(c)

    best_norm: str | None = None
    best_score_sum = -1.0
    for norm, group in by_norm.items():
        s = sum(_score(x) for x in group)
        if s > best_score_sum:
            best_score_sum = s
            best_norm = norm

    assert best_norm is not None
    group = by_norm[best_norm]
    n = len(group)
    confidence_final = sum(_score(x) for x in group) / n if n else 0.0
    votes = len([c for c in cands if c.plate_text_norm == best_norm])
    consistency = votes / len(cands) if cands else 0.0
    needs_review = confidence_final < min_confidence or consistency < 0.5
    if confidence_final < review_confidence:
        needs_review = True

    return FinalPlateDecision(
        plate_text_norm=best_norm,
        confidence_final=min(1.0, confidence_final),
        needs_review=needs_review,
        winning_vote_count=votes,
        total_candidates=len(cands),
    )
