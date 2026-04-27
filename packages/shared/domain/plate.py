"""License plate text normalization and format validation.

Two distinct concepts live here, and they used to be conflated:

  * Known state codes  – every valid Indian state / UT (used by the OCR
                         confusion corrector to fix HR→HE, DL→OL, etc.).
  * Allowed state codes – the deployment-specific allow-list (HR / DL /
                         CH / UP for TSRS Aravali). Used to filter what
                         the API actually ingests, never to reject OCR
                         corrections.

Mixing them broke recognition: if OCR misread "HR" as "HE", the corrector
couldn't map it back because "HE" wasn't even a known state, and the
fallback path simply emitted "HE55A1234" as a "valid" plate.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("plate.domain")

# Default: alphanumeric plates, typical school bus / regional plates (tune per deployment)
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,12}$")

# Common Indian layouts (ST 12 AB 1234, ST 12 1234)
_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")

# Indian BH series (e.g. 21BH1234AA)
_BH_PLATE_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# ---------------------------------------------------------------------------
# Every valid Indian state / UT code (used to correct OCR state-code errors).
# Always the full set — never narrowed by the deployment filter, otherwise
# the corrector loses the ability to map "HE" → "HR" when the deployment
# only allows "HR".
# ---------------------------------------------------------------------------
_KNOWN_STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TG", "TR", "TS", "UK",
    "UP", "WB",
})

# ---------------------------------------------------------------------------
# Deployment-specific allow-list (filters which plates the worker actually
# posts to the API). At TSRS Aravali we only ever see HR / DL / CH / UP.
#
# Override at runtime:
#   PLATE_ALLOWED_STATES=HR,DL,CH,UP   (default)
#   PLATE_ALLOWED_STATES=*             (accept all Indian codes)
#
# Recognition is *not* gated on this set. The validator always tries every
# correction path; this allow-list is consulted by `is_state_allowed()` so
# the caller can decide whether to keep or drop the read.
# ---------------------------------------------------------------------------
_DEFAULT_ALLOWED = frozenset({"HR", "DL", "CH", "UP"})


def _read_allowed_states_from_env() -> frozenset[str]:
    raw = (os.environ.get("PLATE_ALLOWED_STATES") or "").strip()
    if not raw:
        return _DEFAULT_ALLOWED
    if raw == "*":
        return _KNOWN_STATE_CODES
    parts = {p.strip().upper() for p in raw.split(",") if p.strip()}
    valid = parts & _KNOWN_STATE_CODES
    return frozenset(valid) if valid else _DEFAULT_ALLOWED


_ALLOWED_STATE_CODES: frozenset[str] = _read_allowed_states_from_env()

# Public alias for backwards compatibility with code that imported the old
# name. New code should call `get_allowed_state_codes()`.
_VALID_STATE_CODES = _ALLOWED_STATE_CODES


def get_allowed_state_codes() -> frozenset[str]:
    """Currently active deployment allow-list of Indian state codes."""
    return _ALLOWED_STATE_CODES


def set_allowed_state_codes(codes: frozenset[str] | set[str] | list[str] | tuple[str, ...]) -> None:
    """Override the allow-list at runtime (used by tests / hot config)."""
    global _ALLOWED_STATE_CODES, _VALID_STATE_CODES
    cleaned = {str(c).strip().upper() for c in codes if str(c).strip()}
    valid = cleaned & _KNOWN_STATE_CODES
    _ALLOWED_STATE_CODES = frozenset(valid) if valid else _DEFAULT_ALLOWED
    _VALID_STATE_CODES = _ALLOWED_STATE_CODES


def is_state_allowed(plate: str) -> bool:
    """True if the plate's state code is in the deployment allow-list."""
    if not plate or len(plate) < 2:
        return False
    return plate[:2].upper() in _ALLOWED_STATE_CODES


# ---------------------------------------------------------------------------
# OCR confusion correction
# ---------------------------------------------------------------------------

# When position should be a DIGIT but OCR returned a letter-lookalike
_LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "B": "8",
    "S": "5",
    "G": "6",
}

# When position should be a LETTER but OCR returned a digit-lookalike
_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "6": "G",
    "2": "Z",
    "4": "A",  # Indian plate fonts: "4" ↔ "A" confusion at distance
}

# Letter-to-letter visual confusions (low-resolution / distant cameras).
# Each key maps to characters it is commonly misread as. Used for
# state-code correction at the first 2 chars of the plate.
_LETTER_VISUAL_ALTS: dict[str, tuple[str, ...]] = {
    "A": ("H", "M", "N", "R"),
    "B": ("D", "P", "R", "8"),
    "C": ("G", "O", "Q"),
    "D": ("B", "L", "O", "P", "0"),
    "E": ("F", "B", "R"),     # HR→HE: R→E confusion at the bottom curve
    "F": ("D", "E", "P", "R"),
    "G": ("C", "O", "Q", "6"),
    "H": ("A", "M", "N", "K"),
    "I": ("J", "L", "T", "1"),
    "J": ("I", "U"),
    "K": ("X", "R", "H"),
    "L": ("I", "J", "T", "1"),
    "M": ("H", "N", "W"),
    "N": ("H", "M"),
    "O": ("C", "D", "G", "L", "Q", "U", "0"),
    "P": ("B", "D", "F", "R"),
    "Q": ("C", "G", "O"),
    "R": ("B", "F", "P", "K", "A", "E"),
    "S": ("5", "Z"),
    "T": ("I", "J", "L"),
    "U": ("J", "O", "V"),
    "V": ("U", "W", "Y"),
    "W": ("M", "V"),
    "X": ("K",),
    "Y": ("V",),
    "Z": ("2", "S"),
}


def _try_state_code_correction(raw_code: str) -> str | None:
    """
    Try to repair a 2-letter state code via single visual-confusion swaps.
    Always checks against the *full* set of known Indian state codes — we
    don't constrain to the deployment allow-list here, otherwise the
    corrector loses the ability to disambiguate similar codes.
    """
    if len(raw_code) != 2:
        return None
    c0, c1 = raw_code[0], raw_code[1]
    alts0 = (c0,) + _LETTER_VISUAL_ALTS.get(c0, ())
    alts1 = (c1,) + _LETTER_VISUAL_ALTS.get(c1, ())
    for a0 in alts0:
        for a1 in alts1:
            code = a0 + a1
            if code in _KNOWN_STATE_CODES:
                return code
    return None


def normalize_plate_text(raw: str) -> str:
    """Uppercase, strip, remove common OCR noise characters."""
    cleaned = raw.strip().upper()
    for char in (" ", "-", ".", ",", "|", "\\", "/"):
        cleaned = cleaned.replace(char, "")
    return cleaned


def is_valid_plate_format(norm: str) -> bool:
    """Return True if normalized text matches the loose alphanumeric pattern."""
    return bool(_PLATE_PATTERN.match(norm))


def is_indian_plate_format(norm: str) -> bool:
    """Stricter check for typical Indian plate layout (reduces OCR garbage on full frames)."""
    return bool(_INDIAN_PLATE_PATTERN.match(norm)) or bool(_BH_PLATE_PATTERN.match(norm))


def _matches_indian(candidate: str) -> bool:
    return bool(_INDIAN_PLATE_PATTERN.match(candidate)) or bool(_BH_PLATE_PATTERN.match(candidate))


def validate_and_correct_indian(norm: str) -> str | None:
    """
    Validate norm as an Indian plate, applying OCR confusion correction.

    Recognition is intentionally permissive — every valid Indian state code
    is accepted so the worker doesn't silently swallow plates from outside
    the deployment allow-list. The deployment filter is applied separately
    via `is_state_allowed()`, allowing the caller to decide what to log
    versus what to ingest.

    Correction layers (applied in order, return on first success):
      1. Fast path — already matches the regex with a known state code.
      2. State-code visual correction — repair OCR confusions in the
         first 2 chars (HE → HR, DL → OL, FO → DL, …).
      3. Digit↔Letter positional correction — fix O↔0, I↔1, etc.
      4. State-code visual correction PLUS digit↔letter on the rest, as
         a last-ditch attempt for noisy reads.
    """
    if not norm:
        return None
    n = len(norm)
    # Real Indian bus/car plates are 5–12 chars after normalization. We
    # used to reject < 6 to force two-line merging, but that also dropped
    # legitimate short single-line reads like "HR1234". Bump down to 5.
    if n < 5 or n > 12:
        return None

    # ── Layer 1: regex match with a known state code ────────────────────
    if _matches_indian(norm) and norm[:2] in _KNOWN_STATE_CODES:
        return norm

    # ── Layer 2: state-code visual correction (preserve the rest) ───────
    if _matches_indian(norm):
        # Regex matched but state code isn't real — try to recover it.
        corrected_state = _try_state_code_correction(norm[:2])
        if corrected_state and corrected_state != norm[:2]:
            candidate = corrected_state + norm[2:]
            if _matches_indian(candidate):
                return candidate
        # No correction available; still return the original since the
        # plate shape is correct. Caller can decide via is_state_allowed.
        return norm

    # ── Layer 3: digit↔letter positional correction ─────────────────────
    prefix_raw = norm[:2]
    prefix = "".join(_DIGIT_TO_LETTER.get(c, c) for c in prefix_raw)

    for suf_len in (4, 3, 2, 1):
        if n - 2 <= suf_len:
            continue
        middle_raw = norm[2 : n - suf_len]
        suffix = "".join(_LETTER_TO_DIGIT.get(c, c) for c in norm[n - suf_len :])
        for dist_len in range(min(2, len(middle_raw)), 0, -1):
            district = "".join(_LETTER_TO_DIGIT.get(c, c) for c in middle_raw[:dist_len])
            series = "".join(_DIGIT_TO_LETTER.get(c, c) for c in middle_raw[dist_len:])
            candidate = prefix + district + series + suffix
            if _matches_indian(candidate):
                return candidate

    # ── Layer 4: state-code visual correction + digit↔letter on the rest ─
    corrected_state = _try_state_code_correction(prefix_raw)
    if corrected_state and corrected_state != prefix:
        rest = norm[2:]
        for suf_len in (4, 3, 2, 1):
            if len(rest) <= suf_len:
                continue
            middle_raw = rest[: len(rest) - suf_len]
            suffix = "".join(_LETTER_TO_DIGIT.get(c, c) for c in rest[len(rest) - suf_len :])
            for dist_len in range(min(2, len(middle_raw)), 0, -1):
                district = "".join(_LETTER_TO_DIGIT.get(c, c) for c in middle_raw[:dist_len])
                series = "".join(_DIGIT_TO_LETTER.get(c, c) for c in middle_raw[dist_len:])
                candidate = corrected_state + district + series + suffix
                if _matches_indian(candidate):
                    return candidate

    return None
