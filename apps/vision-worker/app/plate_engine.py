"""Plate reads: default VISION_STACK=sample uses YOLO+OpenCV+EasyOCR (sample.txt); rapid uses RapidOCR.

OSD exclusion: Hikvision cameras burn timestamp (top ~10%) and camera name/channel (bottom ~10%)
directly onto the video frame. All ROI extraction paths skip these zones, and fullframe OCR
paths mask them to black before running recognition.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from packages.shared.domain.plate import (
    is_valid_plate_format,
    normalize_plate_text,
    validate_and_correct_indian,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger("vision.plate")

# ---------------------------------------------------------------------------
# Hikvision OSD burn-in zone constants — shared with plate_detection.py
# ---------------------------------------------------------------------------
_OSD_TOP_FRAC = 0.10   # top 10% = timestamp / date overlay (Hikvision typically ~6–8%)
_OSD_BOT_FRAC = 0.16   # bottom 16% = camera name / channel ID (Hikvision typically ~12–15% from bottom)


def _mask_osd_zones(frame: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
    """Return a copy with OSD strips blacked out for fullframe OCR passes."""
    masked = frame.copy()
    fh = masked.shape[0]
    top_h = max(1, int(fh * _OSD_TOP_FRAC))
    bot_h = max(1, int(fh * _OSD_BOT_FRAC))
    masked[:top_h, :] = 0
    masked[fh - bot_h:, :] = 0
    return masked


def _read_plates_sample_stack(
    frame: "NDArray[np.uint8]",
    *,
    min_confidence: float,
    strict_indian: bool,
) -> list[tuple[str, float]]:
    from apps.vision_worker.app.sample_stack import read_plates_sample_stack

    return read_plates_sample_stack(
        frame,
        min_confidence=min_confidence,
        strict_indian=strict_indian,
    )

_engine = None
_MORPH = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def _get_ocr():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        log.info("RapidOCR engine loaded")
    return _engine


def _is_plate_like_region(width: int, height: int, area: float, *, small_mode: bool = False) -> bool:
    """
    Accepts rectangular regions that look like license plates.
    small_mode=True: relaxed constraints for distant/small plates (< 80px wide).
    """
    if height == 0:
        return False
    # Smaller minimum when scanning for distant plates
    min_w = 22 if small_mode else 38
    min_h = 10 if small_mode else 14
    if width < min_w or height < min_h:
        return False
    aspect = width / height
    # Wider aspect range covers angled views and damaged plates
    lo, hi = (1.2, 9.0) if small_mode else (1.6, 8.0)
    if not (lo <= aspect <= hi):
        return False
    # Contour fill ratio — small plates have sparser contours
    bbox_area = width * height
    min_fill = 0.12 if small_mode else 0.20
    if bbox_area > 0 and area / bbox_area < min_fill:
        return False
    return True


def _enhance_roi_for_ocr(roi: "NDArray[np.uint8]", *, target_width: int = 320) -> "NDArray[np.uint8]":
    """
    Upscale + CLAHE only — matches sample.txt fast-mode preprocessing.
    NO binarization: RapidOCR and EasyOCR both need natural-looking images,
    not Otsu black-and-white. Binarization kills deep-learning OCR accuracy.
    """
    if roi is None or roi.size == 0:
        return roi
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
        h, w = gray.shape[:2]

        if w < target_width:
            scale = target_width / w
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # CLAHE only — same as sample.txt _smart_preprocess fast mode
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)
    except Exception:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi


def _roi_score(x: int, y: int, w: int, h: int, fw: int, fh: int) -> float:
    """
    Score ROI candidates so we prioritize diverse sizes rather than just largest area.
    Rewards: plate-like aspect ratio, proximity to lower half of frame (where plates appear),
    and penalizes extremely large (background) or extremely tiny regions.
    Also hard-penalises OSD zones (top/bottom 10%) to ensure they never win over real plates.
    """
    # Hard exclude OSD band — return minimum score so these are never picked
    cy = y + h / 2
    if cy < fh * _OSD_TOP_FRAC or cy > fh * (1.0 - _OSD_BOT_FRAC):
        return -1.0  # Will be filtered out in dedup loop

    aspect = w / max(h, 1)
    ideal_aspect = 4.5  # typical Indian plate ~520x110mm
    aspect_score = 1.0 - min(abs(aspect - ideal_aspect) / ideal_aspect, 1.0)

    # Prefer lower half of frame (plates near gate are in the bottom 60%)
    pos_score = cy / fh  # 0 at top, 1 at bottom

    # Size score: plateau between 60px–400px wide, penalty outside
    size_score = min(w / 80, 1.0) if w < 80 else max(0.0, 1.0 - (w - 400) / 600)

    return 0.4 * aspect_score + 0.35 * pos_score + 0.25 * size_score


def _iter_opencv_plate_rois(
    frame: "NDArray[np.uint8]",
    *,
    max_rois: int = 14,
) -> list[tuple[int, int, int, int, "NDArray[np.uint8]"]]:
    """
    Extract plate-candidate ROIs using edge contours.
    Uses two passes:
      1. Normal scale — catches close-up plates.
      2. 1.8× upscaled frame — catches small/distant plates that are below the pixel threshold.
    ROIs are ranked by _roi_score() so we get a diverse, high-quality set.

    OSD zones (top/bottom 10%) are excluded so Hikvision timestamp / camera-name
    burn-in regions are never passed to OCR.
    """
    fh, fw = frame.shape[:2]
    candidates: list[tuple[float, int, int, int, int]] = []  # (score, x, y, w, h)

    def _extract_contours(img: "NDArray[np.uint8]", scale: float = 1.0) -> None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Slightly wider Canny range catches faint plate borders
        edges = cv2.Canny(gray, 25, 180)
        edges = cv2.dilate(edges, _MORPH, iterations=1)
        edges = cv2.erode(edges, _MORPH, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ih, iw = img.shape[:2]
        small_mode = (scale > 1.0)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if not _is_plate_like_region(w, h, area, small_mode=small_mode):
                continue
            # Map coords back to original frame space
            ox, oy = int(x / scale), int(y / scale)
            ow, oh = max(1, int(w / scale)), max(1, int(h / scale))
            if ox < 0 or oy < 0 or ox + ow > fw or oy + oh > fh:
                continue
            # Skip Hikvision OSD burn-in strips before scoring
            if (oy + oh / 2) < fh * _OSD_TOP_FRAC:
                continue
            if (oy + oh / 2) > fh * (1.0 - _OSD_BOT_FRAC):
                continue
            score = _roi_score(ox, oy, ow, oh, fw, fh)
            if score < 0:
                continue
            candidates.append((score, ox, oy, ow, oh))

    # Pass 1: normal scale
    _extract_contours(frame, scale=1.0)

    # Pass 2: upscaled frame to catch small/distant plates
    scale2 = 1.8
    upscaled = cv2.resize(frame, (int(fw * scale2), int(fh * scale2)), interpolation=cv2.INTER_LINEAR)
    _extract_contours(upscaled, scale=scale2)

    # Deduplicate overlapping regions (keep highest-score)
    candidates.sort(key=lambda c: c[0], reverse=True)
    kept: list[tuple[float, int, int, int, int]] = []
    for cand in candidates:
        sc, cx, cy, cw, ch = cand
        overlap = False
        for _, kx, ky, kw, kh in kept:
            ix = max(cx, kx); iy = max(cy, ky)
            ix2 = min(cx + cw, kx + kw); iy2 = min(cy + ch, ky + kh)
            if ix2 > ix and iy2 > iy:
                inter = (ix2 - ix) * (iy2 - iy)
                union = cw * ch + kw * kh - inter
                if union > 0 and inter / union > 0.4:
                    overlap = True
                    break
        if not overlap:
            kept.append(cand)
        if len(kept) >= max_rois:
            break

    out: list[tuple[int, int, int, int, "NDArray[np.uint8]"]] = []
    for _sc, x, y, w, h in kept:
        roi = frame[y: y + h, x: x + w]
        if roi.size == 0:
            continue
        enhanced = _enhance_roi_for_ocr(roi)
        if enhanced is None or enhanced.size == 0:
            continue
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        out.append((x, y, w, h, bgr))

    return out


def _merge_two_line_plates(result: list) -> list:
    """
    Reconstructs two-line and fragmented plates from RapidOCR output.

    Indian buses have two-line plates AND RapidOCR often fragments each line
    into multiple OCR boxes (e.g. "HR", "26", "BF", "4321" = 4 boxes, 2 rows).

    Strategy:
      1. Cluster all OCR boxes into horizontal rows by Y-centre proximity.
      2. Within each row, sort left-to-right and concatenate text.
      3. Try merging pairs of adjacent rows: if the combined text is a valid
         plate (alphanumeric, 6-12 chars), inject it as an extra candidate.
    """
    if not result or len(result) < 2:
        return result

    def _pts(line: list) -> list:
        try:
            pts = line[0]
            # RapidOCR may return numpy array or list
            if hasattr(pts, "tolist"):
                pts = pts.tolist()
            return pts if pts else []
        except Exception:
            return []

    def _cy(line: list) -> float:
        pts = _pts(line)
        if not pts:
            return 0.0
        return sum(float(p[1]) for p in pts) / len(pts)

    def _cx(line: list) -> float:
        pts = _pts(line)
        if not pts:
            return 0.0
        return sum(float(p[0]) for p in pts) / len(pts)

    def _height(line: list) -> float:
        pts = _pts(line)
        if not pts:
            return 10.0
        ys = [float(p[1]) for p in pts]
        return max(max(ys) - min(ys), 1.0)

    # Sort all boxes top-to-bottom
    sorted_lines = sorted(result, key=_cy)

    # Median height → clustering radius (boxes within 1.5× median height share a row)
    heights = [_height(l) for l in sorted_lines]
    med_h = sorted(heights)[len(heights) // 2] if heights else 20.0
    cluster_radius = max(med_h * 1.5, 8.0)

    # --- Step 1: cluster into horizontal rows ---
    rows: list[list] = []
    for line in sorted_lines:
        cy = _cy(line)
        placed = False
        for row in rows:
            row_cy = sum(_cy(l) for l in row) / len(row)
            if abs(cy - row_cy) < cluster_radius:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    if len(rows) < 2:
        return result  # single row — nothing to merge across lines

    # --- Step 2: within each row, sort L→R and concat text ---
    row_texts: list[tuple[str, float]] = []
    for row in rows:
        row.sort(key=_cx)
        text = "".join(normalize_plate_text(str(l[1]).strip()) for l in row)
        conf = sum(float(l[2]) if len(l) >= 3 else 0.5 for l in row) / len(row)
        row_texts.append((text, conf))

    # --- Step 3: try merging adjacent rows into valid plates ---
    extra = list(result)
    dummy_bbox = result[0][0] if result and result[0] else []
    for i in range(len(row_texts) - 1):
        for j in range(i + 1, min(i + 3, len(row_texts))):
            ta, ca = row_texts[i]
            tb, cb = row_texts[j]
            combined = ta + tb
            avg_conf = (ca + cb) / 2.0
            if 6 <= len(combined) <= 12 and is_valid_plate_format(combined):
                log.info("two-line merge: '%s' + '%s' → '%s' (conf=%.2f)",
                         ta, tb, combined, avg_conf)
                extra.append([dummy_bbox, combined, avg_conf])

    return extra


def _parse_rapid_result(
    result: list | None,
    *,
    min_confidence: float,
    strict_indian: bool,
    debug_label: str = "",
) -> list[tuple[str, float]]:
    if not result:
        return []

    # Try merging two-line plate reads before parsing
    result = _merge_two_line_plates(result)

    out: list[tuple[str, float]] = []
    for line in result:
        try:
            if len(line) < 3:
                continue
            text_raw = str(line[1]).strip()
            score = float(line[2])
        except (IndexError, TypeError, ValueError):
            continue
        norm = normalize_plate_text(text_raw)
        if len(norm) < 4:
            log.info("ocr%s skip '%s' — too short after normalize", debug_label, text_raw)
            continue
        if not is_valid_plate_format(norm):
            log.info("ocr%s skip '%s' — not alphanumeric plate format", debug_label, norm)
            continue
        if strict_indian:
            corrected = validate_and_correct_indian(norm)
            if corrected is None:
                log.info("ocr%s REJECT '%s' conf=%.2f — failed indian validation", debug_label, norm, score)
                continue
            norm = corrected
        if score < min_confidence:
            log.info("ocr%s REJECT '%s' conf=%.2f — below threshold %.2f", debug_label, norm, score, min_confidence)
            continue
        log.info("ocr%s ACCEPT '%s' conf=%.2f", debug_label, norm, score)
        out.append((norm, min(1.0, score)))
    best: dict[str, float] = {}
    for p, s in out:
        best[p] = max(best.get(p, 0.0), s)
    return [(p, best[p]) for p in sorted(best)]


# ---------------------------------------------------------------------------
# Vehicle-first detection: detect bus/car → crop → contrast-boost → OCR
# ---------------------------------------------------------------------------

def _enhance_vehicle_crop(crop: "NDArray[np.uint8]", *, target_min_width: int = 640) -> "NDArray[np.uint8]":
    """
    Enhance a vehicle crop for OCR: upscale + CLAHE contrast + sharpen.
    This is the "increase contrast and perform OCR" step from sample.txt.
    Works on the full colour image (LAB space CLAHE) so RapidOCR's
    DBNet text detector sees natural-looking input.
    """
    h, w = crop.shape[:2]

    # 1. Upscale small crops (distant buses) so OCR has enough pixels
    if w < target_min_width:
        scale = target_min_width / w
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # 2. CLAHE on L channel (LAB) — boosts plate text contrast without colour shift
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 3. Unsharp mask — recovers detail lost to motion blur
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
    enhanced = cv2.addWeighted(enhanced, 2.0, blur, -1.0, 0)

    return enhanced


def _detect_vehicles_and_read_plates(
    frame: "NDArray[np.uint8]",
    *,
    min_confidence: float,
    strict_indian: bool,
) -> list[tuple[str, float]]:
    """
    Vehicle-first plate reading (sample.txt approach):
      1. YOLO detects buses/cars/trucks in the frame
      2. For each vehicle, crop the bounding box
      3. Enhance contrast (CLAHE + sharpen)
      4. Run RapidOCR on the enhanced crop
      5. Parse and validate any plate text found

    Returns list of (plate_text, confidence) or empty list if no vehicles / no plates.
    """
    from apps.vision_worker.app.plate_detection import get_vehicle_detector

    vehicles = get_vehicle_detector().detect(frame)
    if not vehicles:
        return []

    ocr = _get_ocr()
    merged: dict[str, float] = {}
    fh, fw = frame.shape[:2]

    # Sort by confidence descending — process most confident vehicles first
    vehicles.sort(key=lambda v: v["confidence"], reverse=True)

    for veh in vehicles[:6]:  # cap at 6 vehicles per frame
        x1, y1, x2, y2 = veh["bbox"]
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 30 or y2 - y1 < 20:
            continue

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            continue

        # Enhance: upscale + CLAHE + sharpen
        enhanced = _enhance_vehicle_crop(vehicle_crop)

        # OCR on the enhanced vehicle crop
        result, _elapsed = ocr(enhanced)
        if result:
            texts = [(str(l[1]).strip(), round(float(l[2]), 3))
                     for l in result if len(l) >= 3]
            log.info(
                "vehicle %s (%d,%d,%d,%d) conf=%.2f OCR texts: %s",
                veh["class_name"], x1, y1, x2, y2, veh["confidence"],
                texts[:12],
            )
        for plate, score in _parse_rapid_result(
            result,
            min_confidence=max(0.08, min_confidence - 0.04),
            strict_indian=strict_indian,
            debug_label=f"[vehicle-{veh['class_name']}]",
        ):
            merged[plate] = max(merged.get(plate, 0.0), score)

        # Also try the lower half of the vehicle crop (plates are usually at bumper level)
        lower_half = vehicle_crop[vehicle_crop.shape[0] // 2:, :]
        if lower_half.size > 0:
            lower_enh = _enhance_vehicle_crop(lower_half, target_min_width=480)
            result2, _ = ocr(lower_enh)
            if result2:
                texts2 = [(str(l[1]).strip(), round(float(l[2]), 3))
                          for l in result2 if len(l) >= 3]
                log.debug("vehicle lower-half texts: %s", texts2[:8])
            for plate, score in _parse_rapid_result(
                result2,
                min_confidence=max(0.08, min_confidence - 0.04),
                strict_indian=strict_indian,
                debug_label=f"[vehicle-lower-{veh['class_name']}]",
            ):
                merged[plate] = max(merged.get(plate, 0.0), score)

    if merged:
        log.info("vehicle-gated plates found: %s", list(merged.keys()))
    else:
        log.debug("vehicle-gated: %d vehicles detected, no plates read", len(vehicles))

    return [(p, merged[p]) for p in sorted(merged)]


def read_plates_from_frame(
    frame: "NDArray[np.uint8]",
    *,
    min_confidence: float,
    strict_indian: bool = True,
    detection_mode: str = "opencv_roi",
    fullframe_fallback: bool = True,
    max_roi: int = 8,
    vision_stack: str | None = None,
) -> list[tuple[str, float]]:
    """
    Three-step pipeline (sample.txt approach):
      1. license_plate_detector.pt  → YOLO finds plate bbox in frame
      2. Colour crop + CLAHE + 3× upscale  → contrast-boosted plate image
      3. RapidOCR  → read text

    If YOLO finds nothing, falls back to OpenCV contour scan + zone scan.
    """
    stack = (vision_stack or "").strip().lower()
    if not stack:
        from apps.vision_worker.app.settings import get_settings
        stack = get_settings().VISION_STACK.strip().lower()

    if stack == "sample":
        try:
            return _read_plates_sample_stack(frame, min_confidence=min_confidence, strict_indian=strict_indian)
        except Exception as e:
            log.warning("sample stack failed, using rapid: %s", e)

    ocr = _get_ocr()
    merged: dict[str, float] = {}

    # ── Step 1: YOLO plate detector ──────────────────────────────────────────
    # license_plate_detector.pt directly finds plate regions.
    # This is the same model+approach that worked in sample.txt.
    try:
        import os
        from pathlib import Path
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _plate_pt = _models_dir / "license_plate_detector.pt"
        if _plate_pt.is_file():
            if not hasattr(read_plates_from_frame, "_plate_yolo"):
                from ultralytics import YOLO as _YOLO
                read_plates_from_frame._plate_yolo = _YOLO(str(_plate_pt))  # type: ignore[attr-defined]
                read_plates_from_frame._plate_yolo.fuse()  # type: ignore[attr-defined]
                log.info("Loaded plate YOLO: %s", _plate_pt.name)

            yolo = read_plates_from_frame._plate_yolo  # type: ignore[attr-defined]
            device = (os.environ.get("YOLO_DEVICE") or "cpu").strip()
            fh, fw = frame.shape[:2]

            # Run YOLO on 2× upscaled frame so distant/small plates cross the detection threshold
            frame_up = cv2.resize(frame, (fw * 2, fh * 2), interpolation=cv2.INTER_LINEAR)
            results = yolo(frame_up, conf=0.03, iou=0.4, device=device, verbose=False)

            if results and results[0].boxes is not None and len(results[0].boxes):
                log.info("plate YOLO: %d boxes found", len(results[0].boxes))
                for box in results[0].boxes:
                    # Coords are in 2× space — map back to original frame
                    x1, y1, x2, y2 = (int(v / 2) for v in box.xyxy[0].cpu().numpy())
                    conf_det = float(box.conf[0].cpu().numpy())

                    # Clamp + skip tiny / OSD
                    x1, y1 = max(0, x1 - 6), max(0, y1 - 4)
                    x2, y2 = min(fw, x2 + 6), min(fh, y2 + 4)
                    if (x2 - x1) < 30 or (y2 - y1) < 10:
                        continue
                    cy = (y1 + y2) / 2
                    if cy < fh * _OSD_TOP_FRAC or cy > fh * (1 - _OSD_BOT_FRAC):
                        continue

                    # Colour crop → CLAHE → 3× upscale
                    crop = frame[y1:y2, x1:x2].copy()
                    try:
                        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
                        crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
                    except Exception:
                        pass
                    h_c, w_c = crop.shape[:2]
                    crop = cv2.resize(crop, (w_c * 3, h_c * 3), interpolation=cv2.INTER_LANCZOS4)

                    result, _ = ocr(crop)
                    if result:
                        texts = [(str(l[1]).strip(), round(float(l[2]), 3)) for l in result if len(l) >= 3]
                        log.info("plate YOLO box (%d,%d,%d,%d) det_conf=%.2f OCR: %s", x1, y1, x2, y2, conf_det, texts)
                    for plate, score in _parse_rapid_result(result, min_confidence=max(0.05, min_confidence - 0.04), strict_indian=strict_indian, debug_label="[yolo-plate]"):
                        merged[plate] = max(merged.get(plate, 0.0), score)
            else:
                log.info("plate YOLO: no boxes this frame")
    except Exception as exc:
        log.warning("plate YOLO error: %s", exc)

    if merged:
        return [(p, merged[p]) for p in sorted(merged)]

    # ── Step 2: OpenCV contour scan (fallback) ───────────────────────────────
    for rx, ry, rw, rh, _ in _iter_opencv_plate_rois(frame, max_rois=max_roi):
        crop = frame[ry:ry + rh, rx:rx + rw].copy()
        if crop.size == 0:
            continue
        scale = max(1.0, 300 / max(rw, 1))
        crop = cv2.resize(crop, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_LANCZOS4)
        result, _ = ocr(crop)
        if result:
            texts = [(str(l[1]).strip(), round(float(l[2]), 3)) for l in result if len(l) >= 3]
            log.info("contour ROI (%d,%d,%d,%d) texts: %s", rx, ry, rw, rh, texts[:8])
        for plate, score in _parse_rapid_result(result, min_confidence=min_confidence, strict_indian=strict_indian, debug_label="[contour]"):
            merged[plate] = max(merged.get(plate, 0.0), score)

    # ── Step 3: Zone scan (last resort) ─────────────────────────────────────
    if not merged:
        _add_fullframe_results(frame, merged, min_confidence, strict_indian)

    return [(p, merged[p]) for p in sorted(merged)]


def _add_fullframe_results(
    frame: "NDArray[np.uint8]",
    merged: dict[str, float],
    min_confidence: float,
    strict_indian: bool,
) -> None:
    """
    Scan the frame for plates using the sample.txt approach:
      - Focus on the gate zone (lower 70% of frame, full width) — skip sky/trees
      - CLAHE-contrast the crop (colour, not binarized)
      - 2× upscale → RapidOCR
      - Log EVERY text read (not just accepted plates) so we can diagnose rejections

    Mutates `merged` in-place.
    """
    ocr = _get_ocr()
    h, w = frame.shape[:2]
    fb_min = max(0.07, min_confidence - 0.04)

    # ── Zone scan (sample.txt style) ──────────────────────────────────────
    # Plates appear in the lower half of the frame for a gate camera.
    # Skip the top 20% (sky/building) and OSD strips.
    osd_top = max(int(h * 0.10), 1)
    gate_top = max(int(h * 0.20), osd_top)  # skip sky/building
    osd_bot = max(int(h * 0.16), 1)
    gate_bot = h - osd_bot

    zones: list[tuple[int, int, int, int, str]] = [
        # (y1, y2, x1, x2, label)
        # Whole gate zone: where the bus/car actually appears
        (gate_top, gate_bot, 0, w, "gate_zone"),
        # Bumper strip: bottom 30% — where plates always live
        (int(h * 0.65), gate_bot, 0, w, "bumper_strip"),
        # Full frame (safety net — catches plates the zone crops miss)
        (osd_top, gate_bot, 0, w, "full"),
    ]

    for y1, y2, x1, x2, label in zones:
        if merged:
            break  # already found something in an earlier zone
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # CLAHE contrast boost (LAB, colour — same as sample.txt fast mode)
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_ch = clahe_obj.apply(l_ch)
            crop = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        except Exception:
            pass  # use raw crop if LAB fails

        # 2× upscale with LANCZOS (sample.txt: "increase contrast and perform OCR")
        up = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                        interpolation=cv2.INTER_LANCZOS4)

        result, _elapsed = ocr(up)

        # Log ALL texts so we can see exactly what RapidOCR reads
        if result:
            all_texts = [(str(l[1]).strip(), round(float(l[2]), 3))
                         for l in result if len(l) >= 3]
            log.info("zone[%s] OCR raw texts: %s", label, all_texts[:30])
        else:
            log.debug("zone[%s] OCR: no text found", label)

        for plate, score in _parse_rapid_result(
            result,
            min_confidence=fb_min,
            strict_indian=strict_indian,
            debug_label=f"[zone-{label}]",
        ):
            merged[plate] = max(merged.get(plate, 0.0), score)


def rapidocr_try_full_frame(
    frame: "NDArray[np.uint8]",
    *,
    min_confidence: float,
    strict_indian: bool,
) -> list[tuple[str, float]]:
    """RapidOCR on resized full frame when EasyOCR / crops miss (e.g. glossy print).
    OSD zones are masked before OCR so Hikvision burn-in text is invisible."""
    ocr = _get_ocr()
    h, w = frame.shape[:2]
    masked = _mask_osd_zones(frame)
    best: dict[str, float] = {}
    floor = max(0.18, float(min_confidence) * 0.72)
    for max_w in (1600, 1280, 960):
        if w <= max_w:
            small = masked
        else:
            sc = max_w / w
            small = cv2.resize(masked, (int(w * sc), int(h * sc)))
        result, _ = ocr(small)
        for plate, score in _parse_rapid_result(
            result,
            min_confidence=floor,
            strict_indian=strict_indian,
        ):
            best[plate] = max(best.get(plate, 0.0), score)
        if best:
            break
    return [(p, best[p]) for p in sorted(best)]


def mock_demo_snapshot_b64(plate: str, max_w: int) -> str:
    """Synthetic JPEG when no RTSP: watermark so UI is not mistaken for real OCR."""
    w = min(max(320, max_w), 720)
    h = max(140, int(w * 0.22))
    img = np.full((h, w, 3), (24, 26, 30), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (70, 78, 88), 2)
    cv2.putText(
        img,
        "MOCK — no camera (set CAMERA_RTSP_URL)",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 155, 165),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        plate,
        (14, h // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (235, 240, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Demo read — not from video OCR",
        (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (110, 115, 125),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def frame_to_jpeg_b64(frame: "NDArray[np.uint8]", max_w: int) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
