"""
License plate stack (sample.txt OCR + geometry), without vehicle gating:
  - Optional YOLO weights that detect *license plates* only (license_plate_detector.pt)
  - OpenCV Canny / contour regions that look like plates (always — no car/truck required)
  - EasyOCR on those plate ROIs; optional full-frame fallback if nothing found

We do NOT run OCR inside COCO vehicle boxes — that required cars in frame and was wrong for plate-only tests.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from packages.shared.domain.plate import (
    is_valid_plate_format,
    normalize_plate_text,
    validate_and_correct_indian,
)

log = logging.getLogger("vision.sample_stack")

_last_detection_log_ts = 0.0

# --- sample.txt Config (lines 50–72) ---
AI_INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MAX_PLATES_PER_FRAME = 8
OCR_MIN_CONFIDENCE_SAMPLE = 0.35

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _plate_model_path() -> Path | None:
    env = (os.environ.get("YOLO_PLATE_MODEL") or "").strip()
    if env:
        p = Path(env)
        return p if p.is_file() else None
    p = _MODELS_DIR / "license_plate_detector.pt"
    return p if p.is_file() else None


class UltraAccuratePlateDetector:
    """Plate-focused detection: optional YOLO *plate* model + OpenCV plate regions (no vehicle detector)."""

    def __init__(self) -> None:
        self.plate_yolo: Any = None
        self._load_detection_models()

    def _load_detection_models(self) -> None:
        if not _YOLO_AVAILABLE or YOLO is None:
            log.warning("ultralytics YOLO not available — OpenCV detection only")
            return
        primary = _plate_model_path()
        if primary is not None:
            try:
                self.plate_yolo = YOLO(str(primary))
                self.plate_yolo.fuse()
                log.info("Loaded plate YOLO model %s", primary.name)
            except Exception as e:
                log.warning("Plate YOLO failed: %s", e)
        else:
            log.info(
                "No license_plate_detector.pt in models/ — using OpenCV plate regions + EasyOCR only "
                "(set YOLO_PLATE_MODEL or add apps/vision-worker/models/license_plate_detector.pt)",
            )

    def detect_plates_ultra_accurate(self, frame: np.ndarray) -> list[dict[str, Any]]:
        # OpenCV runs always so tests work without vehicles and without a .pt file.
        from apps.vision_worker.app.plate_detection import collect_plate_region_candidates

        from apps.vision_worker.app.settings import get_settings as _gs

        all_detections: list[dict[str, Any]] = list(
            collect_plate_region_candidates(
                frame,
                max_candidates=16,
                use_heuristic_bands=bool(_gs().PLATE_USE_HEURISTIC_BANDS),
            )
        )

        if self.plate_yolo is not None:
            all_detections.extend(self._yolo_plate_detection(frame, self.plate_yolo))

        merged = self._merge_detections(all_detections)
        validated = self._validate_detections(frame, merged)
        return validated[:MAX_PLATES_PER_FRAME]

    def _yolo_plate_detection(self, frame: np.ndarray, model: Any) -> list[dict[str, Any]]:
        """Only plate-trained weights — boxes are plate regions, not cars/trucks."""
        try:
            device = (os.environ.get("YOLO_DEVICE") or "cpu").strip()
            use_half = device.startswith("cuda") and os.environ.get("YOLO_HALF", "").lower() in (
                "1",
                "true",
                "yes",
            )
            try:
                imgsz = int(os.environ.get("YOLO_IMGSZ", str(AI_INPUT_SIZE)))
            except ValueError:
                imgsz = AI_INPUT_SIZE
            imgsz = max(256, min(1280, imgsz))
            results = model(
                frame,
                imgsz=imgsz,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_THRESHOLD,
                device=device,
                half=use_half,
                verbose=False,
            )
            detections: list[dict[str, Any]] = []
            if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
                return detections
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                if not self._validate_bbox_geometry(x1, y1, x2, y2, frame.shape):
                    continue
                roi = frame[int(y1) : int(y2), int(x1) : int(x2)]
                enhanced_roi = self._enhance_roi_for_ocr(roi)
                if enhanced_roi is None:
                    continue
                roi_bgr = (
                    cv2.cvtColor(enhanced_roi, cv2.COLOR_GRAY2BGR)
                    if len(enhanced_roi.shape) == 2
                    else enhanced_roi
                )
                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "roi": roi_bgr,
                        "method": "yolo_plate",
                        "quality_score": self._calculate_roi_quality(enhanced_roi),
                    }
                )
            return detections
        except Exception as e:
            log.debug("yolo_plate_detection: %s", e)
            return []

    def _enhance_roi_for_ocr(self, roi: np.ndarray | None) -> np.ndarray | None:
        if roi is None or roi.size == 0:
            return None
        try:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            height, width = gray.shape
            if width < 300:
                scale = 300 / width
                gray = cv2.resize(
                    gray,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return enhanced
        except Exception:
            return roi

    def _calculate_roi_quality(self, roi: np.ndarray | None) -> float:
        if roi is None or roi.size == 0:
            return 0.0
        try:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            height, width = gray.shape
            size_score = min(1.0, (width * height) / 10000)
            contrast = gray.std()
            contrast_score = min(1.0, contrast / 50)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            edge_score = min(1.0, edge_density * 10)
            aspect_ratio = width / height if height > 0 else 0
            aspect_score = 1.0 if 2.0 <= aspect_ratio <= 5.0 else 0.5
            return size_score * 0.3 + contrast_score * 0.3 + edge_score * 0.3 + aspect_score * 0.1
        except Exception:
            return 0.0

    def _validate_bbox_geometry(
        self, x1: float, y1: float, x2: float, y2: float, frame_shape: tuple[int, ...]
    ) -> bool:
        height, width = frame_shape[:2]
        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            return False
        w, h = x2 - x1, y2 - y1
        if w < 40 or h < 15:
            return False
        if w > width * 0.8 or h > height * 0.8:
            return False
        aspect_ratio = w / h if h > 0 else 0
        return 1.5 <= aspect_ratio <= 8.0

    def _merge_detections(self, all_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all_detections:
            return []
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)
        merged: list[dict[str, Any]] = []
        for detection in all_detections:
            is_duplicate = False
            for existing in merged:
                if self._calculate_iou(detection["bbox"], existing["bbox"]) > 0.5:
                    if detection["confidence"] > existing["confidence"]:
                        merged.remove(existing)
                        merged.append(detection)
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(detection)
        return merged

    def _calculate_iou(self, bbox1: tuple[int, ...], bbox2: tuple[int, ...]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _validate_detections(
        self, frame: np.ndarray, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        validated: list[dict[str, Any]] = []
        for detection in detections:
            if detection["quality_score"] < 0.08:  # was 0.17 — lowered to catch small/distant plates
                continue
            x1, y1, x2, y2 = detection["bbox"]
            w, h = x2 - x1, y2 - y1
            if w < 28 or h < 10:  # was 50/18 — lowered for distant plates
                continue
            validated.append(detection)
        return validated


_detector: UltraAccuratePlateDetector | None = None
_easyocr_reader: Any = None
_init_lock = __import__("threading").Lock()


def get_detector() -> UltraAccuratePlateDetector:
    global _detector
    if _detector is not None:
        return _detector
    with _init_lock:
        if _detector is None:
            _detector = UltraAccuratePlateDetector()
        return _detector


def get_easyocr_reader(gpu: bool | None = None):
    """sample.txt AdaptiveOCRProcessor: EasyOCR Reader(['en'], gpu=…)."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    with _init_lock:
        if _easyocr_reader is not None:
            return _easyocr_reader
    import easyocr

    if gpu is None:
        try:
            from apps.vision_worker.app.settings import get_settings

            gpu = bool(get_settings().OCR_GPU)
        except Exception:
            gpu = os.environ.get("OCR_GPU", "false").lower() in ("1", "true", "yes")
    try:
        _easyocr_reader = easyocr.Reader(["en"], gpu=gpu)
        log.info("EasyOCR loaded (gpu=%s)", gpu)
    except Exception:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        log.info("EasyOCR loaded (cpu fallback)")
    return _easyocr_reader


def _fast_text_cleaning(text: str) -> str | None:
    """Strip to A-Z0-9; allow 8+ chars for Indian HSRP-style 10-character plates."""
    if not text:
        return None
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text if len(text) >= 8 else (text if len(text) >= 6 else None)


_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _bbox_left_x(bbox: Any) -> float:
    arr = np.asarray(bbox, dtype=np.float64)
    return float(arr[:, 0].min())


def _easyocr_read_roi(
    roi: np.ndarray, reader: Any, *, ocr_min: float
) -> tuple[str | None, float]:
    """EasyOCR with allowlist + left-to-right merge (plates split into multiple boxes)."""
    if roi is None or roi.size == 0:
        return None, 0.0
    try:
        frag_floor = max(0.08, ocr_min * 0.42)
        try:
            results = reader.readtext(
                roi,
                detail=1,
                paragraph=False,
                width_ths=0.42,
                height_ths=0.42,
                allowlist=_OCR_ALLOWLIST,
            )
        except TypeError:
            results = reader.readtext(
                roi,
                detail=1,
                paragraph=False,
                width_ths=0.42,
                height_ths=0.42,
            )
        if not results:
            return None, 0.0

        best_text: str | None = None
        best_conf = 0.0
        for _bbox, text, confidence in results:
            if confidence <= ocr_min:
                continue
            clean = _fast_text_cleaning(text)
            if clean and confidence > best_conf:
                best_text = clean
                best_conf = float(confidence)

        parts: list[tuple[str, float]] = []
        for bbox, text, confidence in sorted(results, key=lambda r: _bbox_left_x(r[0])):
            if confidence < frag_floor:
                continue
            t = re.sub(r"[^A-Z0-9]", "", str(text).upper())
            if t:
                parts.append((t, float(confidence)))
        if len(parts) >= 2:
            merged_raw = "".join(p[0] for p in parts)
            merged_clean = _fast_text_cleaning(merged_raw)
            avg_conf = sum(p[1] for p in parts) / len(parts)
            if merged_clean and avg_conf >= frag_floor and (
                best_text is None or len(merged_clean) > len(best_text)
            ):
                return merged_clean, min(0.99, avg_conf)

        return best_text, best_conf
    except Exception as e:
        log.debug("easyocr read: %s", e)
        return None, 0.0


def read_plates_sample_stack(
    frame: np.ndarray,
    *,
    min_confidence: float,
    strict_indian: bool,
) -> list[tuple[str, float]]:
    """
    1) Plate *detection* (regions) — plate_detection + optional plate YOLO.
    2) Plate *recognition* (EasyOCR) — only if PLATE_STAGE=recognition and quality gate passes.
    """
    global _last_detection_log_ts

    from apps.vision_worker.app.settings import get_settings

    s = get_settings()
    stage = (s.PLATE_STAGE or "recognition").strip().lower()
    q_ocr_min = float(s.PLATE_DETECT_MIN_QUALITY_FOR_OCR)

    det = get_detector()
    dets = det.detect_plates_ultra_accurate(frame)

    if stage == "detection":
        now = time.time()
        if now - _last_detection_log_ts >= 3.0:
            _last_detection_log_ts = now
            best_q = max((float(d.get("quality_score") or 0) for d in dets), default=0.0)
            by_method: dict[str, int] = {}
            for d in dets:
                m = str(d.get("method", "?"))
                by_method[m] = by_method.get(m, 0) + 1
            log.info(
                "PLATE_STAGE=detection | regions=%s best_q=%.2f by_method=%s — OCR/API reads off. "
                "When regions look good, set PLATE_STAGE=recognition and tune PLATE_DETECT_MIN_QUALITY_FOR_OCR.",
                len(dets),
                best_q,
                by_method,
            )
        return []

    reader = get_easyocr_reader()
    # Lower floor so EasyOCR reads in the 0.15–0.25 range (common for real plates) aren't thrown away.
    ocr_floor = max(float(min_confidence), 0.15)

    merged: dict[str, float] = {}
    _rejected_reads: list[str] = []  # diagnostic: collect rejected texts

    def _push(norm_key: str, ocr_conf: float, quality: float, *, source: str = "roi") -> None:
        if not is_valid_plate_format(norm_key):
            _rejected_reads.append(f"{source}:{norm_key}(invalid_fmt)")
            return
        if strict_indian:
            corrected = validate_and_correct_indian(norm_key)
            if corrected is None:
                _rejected_reads.append(f"{source}:{norm_key}(not_indian)")
                return
            norm_key = corrected  # use OCR-corrected plate text
        combined = min(1.0, 0.5 * quality + 0.5 * ocr_conf)
        merged[norm_key] = max(merged.get(norm_key, 0.0), combined)

    # ── PRIMARY: two-stage detection (vehicle → plate → enhance → OCR) ────
    # Stage 1: KITTI model detects vehicles (bus/car/truck)
    # Stage 2: license_plate_detector.pt finds the PLATE within the vehicle crop
    # Stage 3: CLAHE + sharpen on just the plate
    # Stage 4: EasyOCR on the isolated, enhanced plate
    try:
        from apps.vision_worker.app.plate_detection import get_vehicle_detector

        vehicles = get_vehicle_detector().detect(frame)
        if vehicles:
            log.info("vehicle-gated: %d vehicles found", len(vehicles))
            fh, fw = frame.shape[:2]
            vehicles.sort(key=lambda v: v["confidence"], reverse=True)

            # Get the plate YOLO model from the already-loaded detector
            plate_yolo = det.plate_yolo  # license_plate_detector.pt

            for veh in vehicles[:6]:
                x1, y1, x2, y2 = veh["bbox"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                if x2 - x1 < 30 or y2 - y1 < 20:
                    continue
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                vw, vh = vehicle_crop.shape[1], vehicle_crop.shape[0]
                log.info(
                    "vehicle %s (%d,%d,%d,%d) conf=%.2f crop=%dx%d",
                    veh["class_name"], x1, y1, x2, y2, veh["confidence"], vw, vh,
                )

                # ── Stage 2: find plate WITHIN the vehicle crop ──────────
                plate_rois_found = False

                # Method A: YOLO plate detector on the vehicle crop
                if plate_yolo is not None:
                    # Upscale vehicle crop so plate detector has more pixels
                    upscale = max(1.0, 640.0 / max(vw, 1))
                    if upscale > 1.0:
                        detect_frame = cv2.resize(
                            vehicle_crop,
                            (int(vw * upscale), int(vh * upscale)),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    else:
                        detect_frame = vehicle_crop

                    try:
                        device = (os.environ.get("YOLO_DEVICE") or "cpu").strip()
                        plate_results = plate_yolo(
                            detect_frame, conf=0.15, iou=0.4,
                            device=device, verbose=False,
                        )
                        if plate_results and hasattr(plate_results[0], "boxes") and plate_results[0].boxes is not None:
                            for pbox in plate_results[0].boxes:
                                px1, py1, px2, py2 = (int(v) for v in pbox.xyxy[0].cpu().numpy())
                                pconf = float(pbox.conf[0].cpu().numpy())

                                # Map back to original vehicle crop coords if upscaled
                                if upscale > 1.0:
                                    px1 = int(px1 / upscale)
                                    py1 = int(py1 / upscale)
                                    px2 = int(px2 / upscale)
                                    py2 = int(py2 / upscale)

                                # Clamp
                                px1, py1 = max(0, px1), max(0, py1)
                                px2, py2 = min(vw, px2), min(vh, py2)

                                plate_roi = vehicle_crop[py1:py2, px1:px2]
                                if plate_roi.size == 0 or plate_roi.shape[1] < 10:
                                    continue

                                plate_rois_found = True
                                log.info(
                                    "  → plate YOLO: (%d,%d,%d,%d) conf=%.2f size=%dx%d",
                                    px1, py1, px2, py2, pconf,
                                    plate_roi.shape[1], plate_roi.shape[0],
                                )

                                # Stage 3: enhance JUST the plate
                                from apps.vision_worker.app.plate_engine import _enhance_vehicle_crop
                                plate_enh = _enhance_vehicle_crop(plate_roi, target_min_width=300)

                                # Stage 4: OCR the isolated plate
                                text, ocr_conf = _easyocr_read_roi(plate_enh, reader, ocr_min=0.10)
                                log.info("    → EasyOCR plate: text=%r conf=%.3f", text, ocr_conf)
                                if text:
                                    norm = normalize_plate_text(text)
                                    if len(norm) >= 4:
                                        _push(norm, ocr_conf, max(0.5, pconf), source="veh_plate")
                    except Exception as e:
                        log.warning("  → plate YOLO on vehicle crop failed: %s", e)

                # Method B: OpenCV plate detection on the vehicle crop (fallback)
                if not plate_rois_found:
                    from apps.vision_worker.app.plate_detection import collect_plate_region_candidates
                    cv_plates = list(collect_plate_region_candidates(
                        vehicle_crop, max_candidates=8, use_heuristic_bands=False,
                    ))
                    if cv_plates:
                        log.info("  → OpenCV found %d plate regions in vehicle crop", len(cv_plates))
                    for cp in cv_plates[:4]:
                        roi = cp.get("roi")
                        if roi is None or roi.size == 0:
                            continue
                        plate_rois_found = True
                        from apps.vision_worker.app.plate_engine import _enhance_vehicle_crop
                        plate_enh = _enhance_vehicle_crop(roi, target_min_width=300)
                        text, ocr_conf = _easyocr_read_roi(plate_enh, reader, ocr_min=0.10)
                        log.info("    → EasyOCR cv-plate: text=%r conf=%.3f", text, ocr_conf)
                        if text:
                            norm = normalize_plate_text(text)
                            if len(norm) >= 4:
                                _push(norm, ocr_conf, float(cp.get("quality_score", 0.4)), source="veh_cv")

                # Method C: if no plate regions found, try lower-third of vehicle
                # (plates on Indian buses are typically at bumper level)
                if not plate_rois_found:
                    lower_third = vehicle_crop[int(vh * 0.6):, :]
                    if lower_third.size > 0:
                        from apps.vision_worker.app.plate_engine import _enhance_vehicle_crop
                        lower_enh = _enhance_vehicle_crop(lower_third, target_min_width=480)
                        text, ocr_conf = _easyocr_read_roi(lower_enh, reader, ocr_min=0.10)
                        log.info("  → EasyOCR lower-third fallback: text=%r conf=%.3f", text, ocr_conf)
                        if text:
                            norm = normalize_plate_text(text)
                            if len(norm) >= 4:
                                _push(norm, ocr_conf, 0.40, source="veh_lower")

            if merged:
                log.info("vehicle-gated TWO-STAGE FOUND: %s", list(merged.keys()))
            else:
                log.warning("vehicle-gated: %d vehicles, plate detector + OCR read ZERO plates", len(vehicles))
        else:
            log.info("vehicle-gated: no vehicles detected in this frame")
    except Exception as e:
        log.warning("vehicle-gated FAILED: %s", e, exc_info=True)

    # ── SUPPLEMENT: existing plate-region scan ────────────────────────────
    # Runs regardless — supplements vehicle-gated results with direct plate ROI reads.
    for d in dets:
        if float(d.get("quality_score") or 0) < q_ocr_min:
            continue
        method = str(d.get("method", ""))
        if method.startswith("heuristic_") and float(d.get("quality_score") or 0) < 0.42:
            continue
        roi = d.get("roi")
        if roi is None:
            continue
        text, ocr_conf = _easyocr_read_roi(roi, reader, ocr_min=ocr_floor)
        if not text:
            continue
        norm = normalize_plate_text(text)
        if len(norm) < 4:
            continue
        if ocr_conf < ocr_floor:
            continue
        _push(norm, ocr_conf, float(d.get("quality_score", 0.5)), source=method[:12])

    # Fullframe fallback — only when ROI pass found nothing (avoids ~5s EasyOCR hit every frame).
    if not merged and bool(getattr(s, "SAMPLE_EASY_OCR_FULLFRAME_FALLBACK", False)):
        from apps.vision_worker.app.plate_detection import mask_osd_zones

        h, w = frame.shape[:2]
        frame_clean = mask_osd_zones(frame)
        ff_min = max(0.12, ocr_floor - 0.06)
        # Single 2× upscale — better for distant small plates, one OCR pass only
        up2 = cv2.resize(frame_clean, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_LANCZOS4)
        text, ocr_conf = _easyocr_read_roi(up2, reader, ocr_min=ff_min)
        if text:
            norm = normalize_plate_text(text)
            if len(norm) >= 4:
                _push(norm, ocr_conf, 0.42, source="fullframe")

    if not merged:
        try:
            from apps.vision_worker.app.plate_detection import mask_osd_zones
            from apps.vision_worker.app.plate_engine import rapidocr_try_full_frame

            # Pass OSD-masked frame so RapidOCR never sees the Hikvision timestamp/camera text
            for plate, sc in rapidocr_try_full_frame(
                mask_osd_zones(frame),
                min_confidence=max(0.12, float(min_confidence) * 0.65),
                strict_indian=strict_indian,
            ):
                merged[plate] = max(merged.get(plate, 0.0), sc)
        except Exception as e:
            log.debug("RapidOCR full-frame fallback: %s", e)

    # Log what we read (and rejected) every ~5 seconds so the dashboard debug panel shows something
    now = time.time()
    if now - _last_detection_log_ts >= 5.0:
        _last_detection_log_ts = now
        log.info(
            "sample_stack: rois=%d merged=%s rejected=%s",
            len(dets),
            list(merged.keys()) or "none",
            _rejected_reads[:8] or "none",
        )

    return [(p, merged[p]) for p in sorted(merged)]
