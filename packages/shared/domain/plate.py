"""License plate text normalization and format validation."""

from __future__ import annotations

import os
import re

# Default: alphanumeric plates, typical school bus / regional plates (tune per deployment)
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,12}$")

# Common Indian layouts (ST 12 AB 1234, ST 12 1234)
_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")

# Indian BH series (e.g. 21BH1234AA)
_BH_PLATE_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# ---------------------------------------------------------------------------
# All valid Indian state / UT codes (used to correct OCR state-code errors)
# ---------------------------------------------------------------------------
_ALL_INDIAN_STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TG", "TR", "TS", "UK",
    "UP", "WB",
})

# ---------------------------------------------------------------------------
# Deployment-restricted state codes
# In TSRS Aravali school deployment we ONLY ever see HR / DL / CH / UP plates
# (Haryana / Delhi / Chandigarh / Uttar Pradesh — all NCR). Restricting the
# validator to this allow-list dramatically reduces false positives because
# the state-code corrector will refuse to "fix" garbage OCR into AP/KA/MH
# plates that physically can't be at this gate.
#
# Override at runtime with PLATE_ALLOWED_STATES env var, e.g.
#   PLATE_ALLOWED_STATES=HR,DL,CH,UP   (default)
#   PLATE_ALLOWED_STATES=*             (accept all Indian codes)
# ---------------------------------------------------------------------------
_DEFAULT_ALLOWED = frozenset({"HR", "DL", "CH", "UP"})


def _read_allowed_states_from_env() -> frozenset[str]:
    raw = (os.environ.get("PLATE_ALLOWED_STATES") or "").strip()
    if not raw:
        return _DEFAULT_ALLOWED
    if raw == "*":
        return _ALL_INDIAN_STATE_CODES
    parts = {p.strip().upper() for p in raw.split(",") if p.strip()}
    valid = parts & _ALL_INDIAN_STATE_CODES
    return frozenset(valid) if valid else _DEFAULT_ALLOWED


_VALID_STATE_CODES: frozenset[str] = _read_allowed_states_from_env()


def get_allowed_state_codes() -> frozenset[str]:
    """Currently active allow-list of Indian state codes."""
    return _VALID_STATE_CODES


def set_allowed_state_codes(codes: frozenset[str] | set[str] | list[str] | tuple[str, ...]) -> None:
    """Override the allow-list at runtime (used by tests / hot config)."""
    global _VALID_STATE_CODES
    cleaned = {str(c).strip().upper() for c in codes if str(c).strip()}
    valid = cleaned & _ALL_INDIAN_STATE_CODES
    _VALID_STATE_CODES = frozenset(valid) if valid else _DEFAULT_ALLOWED

# ---------------------------------------------------------------------------
# OCR confusion correction
# EasyOCR / RapidOCR frequently swap visually similar characters.
# Two correction paths:
#   1. Digit ↔ Letter swaps at expected-type positions (existing logic)
#   2. Letter ↔ Letter visual confusions at the STATE CODE position (new)
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
# Each key maps to characters it is commonly misread as.
# Used for state-code correction only (first 2 chars of plate).
_LETTER_VISUAL_ALTS: dict[str, tuple[str, ...]] = {
    "A": ("H", "M", "N"),
    "B": ("D", "P", "R", "8"),
    "C": ("G", "O", "Q"),
    "D": ("B", "L", "O", "P", "0"),
    "E": ("F",),
    "F": ("D", "E", "P", "R"),    # DL→FO: D→F is the key confusion
    "G": ("C", "O", "Q", "6"),
    "H": ("A", "M", "N"),
    "I": ("J", "L", "T", "1"),
    "J": ("I", "U"),
    "K": ("X", "R"),
    "L": ("I", "J", "T", "1"),
    "M": ("H", "N", "W"),
    "N": ("H", "M"),
    "O": ("C", "D", "G", "L", "Q", "U", "0"),  # DL→FO: L→O is the key confusion
    "P": ("B", "D", "F", "R"),
    "Q": ("C", "G", "O"),
    "R": ("B", "F", "P"),
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
    Given the first 2 chars OCR read, find the nearest valid Indian state code.
    Tries single-character substitutions using visual confusion alternatives.
    Returns the corrected 2-letter state code, or None if no valid code found.
    """
    if len(raw_code) != 2:
        return None
    c0, c1 = raw_code[0], raw_code[1]
    # Try all combinations: original or any visual alternative for each char
    alts0 = (c0,) + _LETTER_VISUAL_ALTS.get(c0, ())
    alts1 = (c1,) + _LETTER_VISUAL_ALTS.get(c1, ())
    for a0 in alts0:
        for a1 in alts1:
            code = a0 + a1
            if code in _VALID_STATE_CODES:
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


def validate_and_correct_indian(norm: str) -> str | None:
    """
    Validate norm as an Indian plate, applying OCR confusion correction if needed.

    Correction layers (applied in order):
      1. Fast path — already matches regex as-is.
      2. Digit↔Letter swaps at expected-type positions (O→0, I→1, 0→O, 1→I, etc.)
      3. State-code visual correction — tries letter-to-letter visual alternatives
         for the first 2 chars against all known Indian state codes (catches DL→FO etc.)

    Returns the (possibly corrected) plate text if valid, None otherwise.
    """
    if not norm:
        return None
    n = len(norm)
    # Real Indian bus/car plates are 6–12 chars (e.g. MH12AB1234 = 10, DL1C1234 = 8).
    # 5-char strings like "HR554" are almost always the first line of a two-line plate
    # that hasn't been merged yet — rejecting them forces the two-line merger to win.
    if n < 6 or n > 12:
        return None

    # Layer 1: fast path — regex match AND valid state code
    if _INDIAN_PLATE_PATTERN.match(norm) and norm[:2] in _VALID_STATE_CODES:
        return norm
    if _BH_PLATE_PATTERN.match(norm):
        return norm

    # Layer 2: digit↔letter positional correction
    # Structure: first 2 = state letters, last 1-4 = serial digits, middle = district+series
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
            if _INDIAN_PLATE_PATTERN.match(candidate) or _BH_PLATE_PATTERN.match(candidate):
                return candidate

    # Layer 3: state-code visual confusion correction (e.g. FO → DL)
    # If layers 1+2 found nothing, try correcting the first 2 chars via visual alternatives
    # then repeat the digit↔letter correction on the rest.
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
                if _INDIAN_PLATE_PATTERN.match(candidate) or _BH_PLATE_PATTERN.match(candidate):
                    return candidate

    return None
