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

from apps.vision_worker.app.draw_overlay import (
    BoxRecord,
    overlay_add_bumper,
    overlay_add_plate_region,
    overlay_add_vehicle,
    overlay_drain,
    overlay_reset,
)
from packages.shared.domain.plate import (
    is_state_allowed,
    is_valid_plate_format,
    looks_like_bus_body_text,
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


# Multiple mirrors — first one that returns a >1 MB body wins. Some
# corporate / school networks block HuggingFace; the GitHub mirror is
# the proven fallback. A custom user-friendly URL can be injected via
# the YOLO_PLATE_MODEL_URL env var (single URL, takes priority).
_PLATE_MODEL_DOWNLOAD_URLS: tuple[str, ...] = (
    "https://huggingface.co/morsetechlab/yolov11-license-plate-detection"
    "/resolve/main/license-plate-finetune-v1n.pt",
    "https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8"
    "/raw/main/license_plate_detector.pt",
)
"""Public single-class yolov11n license-plate detector. ~5-6 MB, single
class "License_Plate". Downloaded on first start if no local weight is
found and baked into the Docker image at build time so the container
also works offline."""


def _try_download_plate_model(target: Path) -> bool:
    """Best-effort fetch of a public license-plate YOLO weight.

    Tries every mirror in `_PLATE_MODEL_DOWNLOAD_URLS` (or a single
    `YOLO_PLATE_MODEL_URL` override) and returns True on the first one
    whose body is >=1 MB. Failures are swallowed — the rest of the
    pipeline still works without a plate detector, just less accurately.
    """
    if target.is_file() and target.stat().st_size > 1_000_000:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)

    override = (os.environ.get("YOLO_PLATE_MODEL_URL") or "").strip()
    urls: tuple[str, ...] = (override,) if override else _PLATE_MODEL_DOWNLOAD_URLS

    import urllib.request

    last_exc: Exception | None = None
    for url in urls:
        log.warning(
            "license_plate_detector.pt not found at %s — trying %s",
            target, url,
        )
        try:
            tmp = target.with_suffix(".pt.tmp")
            # Some CDNs (HuggingFace, GitHub raw) 403 default urllib UA.
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SchoolVF/1.0)",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 1_000_000:
                log.warning(
                    "plate model download too small (%d bytes) from %s — trying next mirror",
                    len(data), url,
                )
                continue
            tmp.write_bytes(data)
            tmp.replace(target)
            log.warning(
                "Plate model downloaded OK: %s (%d bytes) from %s",
                target, target.stat().st_size, url,
            )
            return True
        except Exception as exc:
            last_exc = exc
            log.warning("Plate model download failed from %s: %s — trying next mirror", url, exc)
            continue

    log.error(
        "All plate model mirrors failed (last error: %s). "
        "MANUAL FIX: download a yolov8/yolov11 license-plate .pt and place it at %s. "
        "Set YOLO_PLATE_MODEL_URL=<your-url> to add a custom mirror, or "
        "YOLO_PLATE_AUTODOWNLOAD=0 to silence this message. "
        "Pipeline still runs (OpenCV contour fallback) — just less accurate.",
        last_exc, target,
    )
    return False


def _plate_model_path() -> Path | None:
    env = (os.environ.get("YOLO_PLATE_MODEL") or "").strip()
    if env:
        p = Path(env)
        return p if p.is_file() else None
    p = _MODELS_DIR / "license_plate_detector.pt"
    if p.is_file():
        return p
    # Auto-download if missing. Skipped when YOLO_PLATE_AUTODOWNLOAD=0.
    if (os.environ.get("YOLO_PLATE_AUTODOWNLOAD", "1") or "1").strip() not in ("0", "false", "no"):
        if _try_download_plate_model(p):
            return p
    return None


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
                # Move to GPU + FP16 + warm up. Without this the FIRST plate
                # detection on a real frame pays a 1-2 second CUDA init cost
                # which is what makes the user wait "8-10 s after the bus
                # comes" the very first time.
                try:
                    device = (os.environ.get("YOLO_DEVICE") or "cpu").strip()
                    half = os.environ.get("YOLO_HALF", "0").lower() in ("1", "true", "yes")
                    try:
                        imgsz = int(os.environ.get("YOLO_IMGSZ", "0") or "0")
                    except ValueError:
                        imgsz = 0
                    if device.startswith("cuda"):
                        self.plate_yolo.to(device)
                        if half:
                            self.plate_yolo.model.half()
                        warm = np.zeros((imgsz or 640, imgsz or 640, 3), dtype=np.uint8)
                        self.plate_yolo(
                            warm, conf=0.5, device=device, half=half,
                            imgsz=imgsz or 640, verbose=False,
                        )
                        log.info(
                            "Plate YOLO warmed up on %s (half=%s imgsz=%s)",
                            device, half, imgsz or 640,
                        )
                except Exception as warm_exc:
                    log.debug("Plate YOLO warmup skipped: %s", warm_exc)
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
    """Strip to A-Z0-9; keep anything 5+ chars so the validator can decide.

    Indian HSRP plates are 8–10 chars, but partial reads like 'HR1234'
    (6 chars) or 'DL3CAB' (6 chars from a half-read two-line plate) are
    real plates after correction. Letting them through here means the
    validator's state-code correction + two-line merge can salvage them.
    """
    if not text:
        return None
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text if len(text) >= 5 else None


_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _candidate_plate_overlay(det_conf: float, *, note: str = "") -> str:
    """Label for orange boxes: detector saw a plate-like region but no plate string was accepted."""
    s = f"no OCR match · det {det_conf:.2f}"
    return s + note


def _bbox_left_x(bbox: Any) -> float:
    arr = np.asarray(bbox, dtype=np.float64)
    return float(arr[:, 0].min())


def _bbox_top_y(bbox: Any) -> float:
    arr = np.asarray(bbox, dtype=np.float64)
    return float(arr[:, 1].min())


def _bbox_height(bbox: Any) -> float:
    arr = np.asarray(bbox, dtype=np.float64)
    return float(arr[:, 1].max() - arr[:, 1].min())


def _stacked_two_line_merge(results: list) -> list[tuple[str, float]]:
    """
    Reconstruct stacked/fragmented Indian plates from EasyOCR results.

    EasyOCR often returns:
      - one token per fragment ("DL", "3CAB", "1234"), or
      - two lines with slight tilt / skew.
    We cluster by row using Y-centres, then merge adjacent rows (2-line and
    3-line cases) with a proximity gate.
    """
    if not results or len(results) < 2:
        return []

    # (y_center, y_height, x_left, text, conf)
    items: list[tuple[float, float, float, str, float]] = []
    for bbox, text, conf in results:
        try:
            t = re.sub(r"[^A-Z0-9]", "", str(text).upper())
            if not t:
                continue
            yt = _bbox_top_y(bbox)
            h = max(_bbox_height(bbox), 1.0)
            yc = yt + (h * 0.5)
            items.append((yc, h, _bbox_left_x(bbox), t, float(conf)))
        except Exception:
            continue
    if len(items) < 2:
        return []

    # Step 1: cluster OCR tokens into horizontal rows.
    items.sort(key=lambda r: r[0])
    rows: list[list[tuple[float, float, float, str, float]]] = []
    for it in items:
        placed = False
        for row in rows:
            row_y = sum(r[0] for r in row) / len(row)
            row_h = max(r[1] for r in row)
            # Use token/row height, not frame-level constants.
            if abs(it[0] - row_y) <= max(row_h * 1.0, it[1] * 1.0, 8.0):
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])
    if len(rows) < 2:
        return []

    # Step 2: normalize each row text (L->R).
    rows.sort(key=lambda row: sum(r[0] for r in row) / len(row))
    row_texts: list[tuple[str, float]] = []
    row_yc: list[float] = []
    row_h: list[float] = []
    for row in rows:
        row.sort(key=lambda r: r[2])
        txt = "".join(r[3] for r in row)
        conf = sum(r[4] for r in row) / max(1, len(row))
        row_texts.append((txt, conf))
        row_yc.append(sum(r[0] for r in row) / len(row))
        row_h.append(max(r[1] for r in row))

    # Step 3: merge adjacent rows with geometric gating.
    out: list[tuple[str, float]] = []
    for i in range(len(row_texts) - 1):
        for j in range(i + 1, min(i + 3, len(row_texts))):
            gap = abs(row_yc[j] - row_yc[i])
            max_h = max(row_h[i], row_h[j], 1.0)
            if gap > max_h * 4.5:
                continue
            a, ca = row_texts[i]
            b, cb = row_texts[j]
            merged = a + b
            if 6 <= len(merged) <= 12:
                out.append((merged, (ca + cb) / 2.0))

    # Rare 3-line split: "HR" / "26BF" / "1234".
    if len(row_texts) >= 3:
        for i in range(len(row_texts) - 2):
            a, ca = row_texts[i]
            b, cb = row_texts[i + 1]
            c, cc = row_texts[i + 2]
            if not (a and b and c):
                continue
            max_h = max(row_h[i], row_h[i + 1], row_h[i + 2], 1.0)
            if (
                abs(row_yc[i + 1] - row_yc[i]) <= max_h * 4.5
                and abs(row_yc[i + 2] - row_yc[i + 1]) <= max_h * 4.5
            ):
                merged = a + b + c
                if 6 <= len(merged) <= 12:
                    out.append((merged, (ca + cb + cc) / 3.0))

    # Deduplicate by text, keep best confidence.
    best: dict[str, float] = {}
    for txt, conf in out:
        best[txt] = max(best.get(txt, 0.0), conf)
    return sorted(best.items(), key=lambda kv: (-len(kv[0]), -kv[1]))


def _easyocr_read_roi(
    roi: np.ndarray, reader: Any, *, ocr_min: float
) -> tuple[str | None, float]:
    """EasyOCR with allowlist + left-to-right merge (plates split into multiple boxes)."""
    if roi is None or roi.size == 0:
        return None, 0.0
    try:
        frag_floor = max(0.055, ocr_min * 0.36)
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
                best_text, best_conf = merged_clean, min(0.99, avg_conf)

        # Stacked two-line plate fallback: top→bottom row concat (DLABC / 1234).
        # Always try this — if it produces a valid Indian-format plate, prefer it.
        for stacked_text, stacked_conf in _stacked_two_line_merge(results):
            cleaned = _fast_text_cleaning(stacked_text)
            if not cleaned:
                continue
            if validate_and_correct_indian(cleaned) is not None and (
                best_text is None or len(cleaned) > len(best_text)
            ):
                return cleaned, min(0.99, stacked_conf)

        return best_text, best_conf
    except Exception as e:
        log.debug("easyocr read: %s", e)
        return None, 0.0


def read_plates_sample_stack(
    frame: np.ndarray,
    *,
    min_confidence: float,
    strict_indian: bool,
) -> tuple[list[tuple[str, float]], list[BoxRecord]]:
    """
    1) Plate *detection* (regions) — plate_detection + optional plate YOLO.
    2) Plate *recognition* (EasyOCR) — only if PLATE_STAGE=recognition and quality gate passes.

    Returns (plates, overlay_boxes). The boxes describe the geometry the
    detector saw (vehicle bbox, bumper crop, plate ROI), with `accepted=True`
    on plate boxes that yielded a successful OCR read. The pipeline draws
    these on the snapshot so the dashboard shows what was actually scanned.
    """
    global _last_detection_log_ts

    # Reset the per-thread overlay bag at the very top of the call so we
    # never leak boxes from a previous frame on the same worker thread.
    overlay_reset()

    from apps.vision_worker.app.settings import get_settings

    s = get_settings()
    stage = (s.PLATE_STAGE or "recognition").strip().lower()
    q_ocr_min = float(s.PLATE_DETECT_MIN_QUALITY_FOR_OCR)
    # Upscale plate crops aggressively — default 720 px min width uses more
    # GPU but reads clear plates far more reliably than 300 px.
    ocr_target_w = max(480, int(getattr(s, "PLATE_OCR_TARGET_MIN_WIDTH", 720)))
    ocr_bumper_w = max(ocr_target_w, 560)
    bumper_top_frac = max(0.20, min(0.55, float(s.VEHICLE_BUMPER_TOP_FRAC)))
    plate_yolo_conf = max(0.02, min(0.45, float(s.PLATE_YOLO_MIN_CONF)))

    det = get_detector()

    # Full-frame plate detection is only needed for the supplement path
    # (when the vehicle-gated path finds nothing) and for PLATE_STAGE=detection.
    # Otherwise we'd waste 50-150 ms on a redundant YOLO call.
    dets: list[dict] = []
    _dets_computed = False

    def _ensure_dets() -> list[dict]:
        nonlocal _dets_computed, dets
        if not _dets_computed:
            dets = det.detect_plates_ultra_accurate(frame)
            _dets_computed = True
        return dets

    if stage == "detection":
        dets = _ensure_dets()
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
        return [], []

    reader = get_easyocr_reader()
    # Lower floor so EasyOCR keeps weak-but-readable fragments; validator filters noise.
    ocr_floor = max(float(min_confidence), 0.07)
    veh_ocr_min = max(0.055, float(min_confidence))

    merged: dict[str, float] = {}
    _rejected_reads: list[str] = []  # diagnostic: collect rejected texts

    def _push(norm_key: str, ocr_conf: float, quality: float, *, source: str = "roi") -> None:
        # First gate: body-paint blacklist. Fastest early-exit and stops
        # 'SCHUOL' / 'ARAVALI' / 'ONDUTY' from ever reaching the validator.
        if looks_like_bus_body_text(norm_key):
            _rejected_reads.append(f"{source}:{norm_key}(bus_body)")
            log.info("plate REJECT %s '%s' — bus body text", source, norm_key)
            return
        if not is_valid_plate_format(norm_key):
            _rejected_reads.append(f"{source}:{norm_key}(invalid_fmt)")
            log.info("plate REJECT %s '%s' — not alphanumeric 4-12", source, norm_key)
            return
        if strict_indian:
            corrected = validate_and_correct_indian(norm_key)
            if corrected is None:
                _rejected_reads.append(f"{source}:{norm_key}(not_indian)")
                log.info("plate REJECT %s '%s' — failed indian validation", source, norm_key)
                return
            if corrected != norm_key:
                log.info("plate correct %s '%s' → '%s'", source, norm_key, corrected)
            norm_key = corrected
            # Deployment allow-list (HR/DL/CH/UP by default) — separate from
            # recognition. Logged at WARNING so it's obvious in the docker logs
            # when valid plates are being rejected because of the filter.
            if not is_state_allowed(norm_key):
                _rejected_reads.append(f"{source}:{norm_key}(state_filtered)")
                log.warning(
                    "plate REJECT %s '%s' — state '%s' not in PLATE_ALLOWED_STATES (set '*' to accept all)",
                    source, norm_key, norm_key[:2],
                )
                return
        combined = min(1.0, 0.5 * quality + 0.5 * ocr_conf)
        merged[norm_key] = max(merged.get(norm_key, 0.0), combined)
        log.info("plate ACCEPT %s '%s' ocr_conf=%.2f q=%.2f → %.2f",
                 source, norm_key, ocr_conf, quality, combined)

    # ── Per-frame OCR budget ──────────────────────────────────────────────
    OCR_BUDGET = max(6, min(16, int(s.PLATE_OCR_BUDGET)))
    ocr_calls = 0

    def _ocr_with_budget(roi_img, *, ocr_min: float) -> tuple[str | None, float]:
        nonlocal ocr_calls
        if ocr_calls >= OCR_BUDGET:
            return None, 0.0
        ocr_calls += 1
        return _easyocr_read_roi(roi_img, reader, ocr_min=ocr_min)

    def _ocr_and_push(roi_img, *, source: str, quality: float, ocr_min: float = 0.10) -> bool:
        """Run OCR on a single ROI and push the read. Returns True on success."""
        text, ocr_conf = _ocr_with_budget(roi_img, ocr_min=ocr_min)
        if not text:
            return False
        norm = normalize_plate_text(text)
        if len(norm) < 4:
            return False
        before = len(merged)
        _push(norm, ocr_conf, quality, source=source)
        return len(merged) > before

    def _ocr_two_line_split_and_push(
        roi_img,
        *,
        source: str,
        quality: float,
        ocr_min: float = 0.07,
    ) -> bool:
        """
        Rescue path for stacked plates: OCR top and bottom halves separately,
        then concatenate. Helps when one EasyOCR pass only reads one line.
        """
        if roi_img is None or getattr(roi_img, "size", 0) == 0:
            return False
        h, w = roi_img.shape[:2]
        if h < 24 or w < 24:
            return False
        # Overlapping split to survive slight vertical misalignment.
        split = int(h * 0.52)
        top = roi_img[: max(1, split), :]
        bottom = roi_img[max(0, int(h * 0.36)) :, :]
        top_text, top_conf = _ocr_with_budget(top, ocr_min=ocr_min)
        bot_text, bot_conf = _ocr_with_budget(bottom, ocr_min=ocr_min)
        if not top_text or not bot_text:
            return False
        combined = normalize_plate_text(top_text + bot_text)
        if len(combined) < 6:
            return False
        before = len(merged)
        _push(
            combined,
            min(0.99, (float(top_conf) + float(bot_conf)) / 2.0),
            quality,
            source=f"{source}_split",
        )
        return len(merged) > before

    def _best_merged_plate_len() -> int:
        return max((len(p) for p in merged), default=0)

    # ── PRIMARY: vehicle YOLO → plate YOLO inside the bus' BOTTOM 50% ─────
    # Plates on Indian school buses live on the front/rear bumper, never on
    # the sides or upper body. By cropping to the bottom 50% of the vehicle
    # bbox before the plate detector runs we (a) make the YOLO call faster
    # (smaller input) and (b) make it physically impossible for body-paint
    # text like "ON SCHOOL DUTY" or "ARAVALI" to be misread as a plate.
    try:
        from apps.vision_worker.app.plate_detection import get_vehicle_detector

        vehicles = get_vehicle_detector().detect(frame)
        if vehicles:
            log.info("vehicle-gated: %d vehicles found", len(vehicles))
            fh, fw = frame.shape[:2]
            # Pick the BIGGEST bbox in frame, not the highest-confidence one —
            # the bus we want to read is the one closest to the camera, which
            # is the one occupying the most pixels. Smaller cars / background
            # vehicles are ignored.
            vehicles.sort(
                key=lambda v: (
                    (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1])
                ),
                reverse=True,
            )

            plate_yolo = det.plate_yolo  # license_plate_detector.pt

            # ONLY scan the closest (biggest) vehicle. The user explicitly
            # described the pipeline as: detect bus → snapshot → plate
            # detector inside the bus → OCR each plate box. Adding more
            # vehicles per frame just creates backlog without finding more
            # of "the" bus.
            for veh in vehicles[:1]:
                if ocr_calls >= OCR_BUDGET or merged:
                    # Either we've already found a plate or the budget is
                    # spent — stop scanning more vehicles this frame.
                    break
                x1, y1, x2, y2 = veh["bbox"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                if x2 - x1 < 30 or y2 - y1 < 20:
                    continue

                # Lower portion of vehicle (bumper/grille); fraction from settings
                # so high plates / loose vehicle boxes still crop the plate band.
                vh_total = y2 - y1
                bumper_top = y1 + int(vh_total * bumper_top_frac)
                bumper_crop = frame[bumper_top:y2, x1:x2]
                if bumper_crop.size == 0:
                    continue
                vw, vh = bumper_crop.shape[1], bumper_crop.shape[0]
                # Collect overlay geometry so the snapshot can show the user
                # exactly which region we cropped + scanned.
                overlay_add_vehicle(
                    (x1, y1, x2, y2),
                    conf=float(veh.get("confidence", 0.0)),
                    label=str(veh.get("class_name") or "veh"),
                )
                overlay_add_bumper((x1, bumper_top, x2, y2))
                log.info(
                    "vehicle %s (%d,%d,%d,%d) conf=%.2f bumper=%dx%d",
                    veh["class_name"], x1, y1, x2, y2, veh["confidence"], vw, vh,
                )

                from apps.vision_worker.app.plate_engine import _enhance_vehicle_crop

                plate_rois_found = False

                # Method A: plate-trained YOLO inside the bumper crop only.
                if plate_yolo is not None:
                    # Match upscale target to YOLO_IMGSZ so we feed the
                    # detector real pixels at its working resolution. For
                    # a tiny 200-px bumper this means 4.8× upscale to 960
                    # — small plates that used to be 12 px tall become
                    # 60 px tall, which YOLO can actually find.
                    try:
                        _imgsz_target = int(os.environ.get("YOLO_IMGSZ", "640") or "640")
                    except ValueError:
                        _imgsz_target = 640
                    upscale = max(1.0, float(_imgsz_target) / max(vw, 1))
                    if upscale > 1.0:
                        detect_frame = cv2.resize(
                            bumper_crop,
                            (int(vw * upscale), int(vh * upscale)),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    else:
                        detect_frame = bumper_crop

                    try:
                        device = (os.environ.get("YOLO_DEVICE") or "cpu").strip()
                        half = os.environ.get("YOLO_HALF", "0").lower() in ("1", "true", "yes")
                        try:
                            imgsz = int(os.environ.get("YOLO_IMGSZ", "0") or "0")
                        except ValueError:
                            imgsz = 0
                        yolo_kwargs = dict(
                            conf=plate_yolo_conf,
                            iou=0.45,
                            device=device,
                            half=half,
                            verbose=False,
                        )
                        if imgsz >= 320:
                            yolo_kwargs["imgsz"] = imgsz
                        plate_results = plate_yolo(detect_frame, **yolo_kwargs)
                        if plate_results and hasattr(plate_results[0], "boxes") and plate_results[0].boxes is not None:
                            # Sort plate boxes by confidence descending so we
                            # spend OCR budget on the best ones first.
                            sorted_boxes = sorted(
                                plate_results[0].boxes,
                                key=lambda b: float(b.conf[0].cpu().numpy()),
                                reverse=True,
                            )[:5]
                            for pbox in sorted_boxes:
                                if ocr_calls >= OCR_BUDGET:
                                    break
                                px1, py1, px2, py2 = (int(v) for v in pbox.xyxy[0].cpu().numpy())
                                pconf = float(pbox.conf[0].cpu().numpy())
                                if upscale > 1.0:
                                    px1 = int(px1 / upscale)
                                    py1 = int(py1 / upscale)
                                    px2 = int(px2 / upscale)
                                    py2 = int(py2 / upscale)
                                px1, py1 = max(0, px1), max(0, py1)
                                px2, py2 = min(vw, px2), min(vh, py2)

                                ph = py2 - py1
                                pw = px2 - px1
                                if pw < 8 or ph < 5:
                                    continue

                                # Padding around the YOLO box. Plate
                                # detectors regularly clip 1-2 chars at
                                # the edges, especially on the first line
                                # of a stacked plate, so we always pad.
                                px1_pad = max(0, px1 - max(4, pw // 20))
                                px2_pad = min(vw, px2 + max(4, pw // 20))

                                # ★ Two-line plate extension: pad the box
                                # downward by one full plate height so a
                                # stacked plate's second line is captured.
                                # We ALSO pad upward by a little to catch
                                # the first-line top edge.
                                py1_pad = max(0, py1 - max(2, ph // 10))
                                py2_ext = min(vh, py2 + ph)

                                plate_rois_found = True
                                # Absolute coordinates of this plate box in
                                # the original frame (for overlay rendering).
                                plate_abs_bbox = (
                                    x1 + px1_pad, bumper_top + py1_pad,
                                    x1 + px2_pad, bumper_top + py2_ext,
                                )
                                log.info(
                                    "  → plate YOLO box (%d,%d,%d,%d) conf=%.2f ratio=%.2f → tight+2line",
                                    px1, py1, px2, py2, pconf, (ph / max(pw, 1)),
                                )

                                # Always do BOTH crops:
                                #   1) the tight (padded) box  → wins on single-line plates
                                #   2) the downward-extended box → wins on stacked 2-line plates
                                # We prefer whichever yields the longer valid plate.
                                # Without this, stacked plates like "HR55B / BC2973"
                                # report only the first line ("HR558") because the
                                # YOLO box's tight crop happens to be readable.

                                # Crop 1: tight (single-line plate winner)
                                merged_before_tight = dict(merged)
                                tight = bumper_crop[py1_pad:py2, px1_pad:px2_pad]
                                if tight.size > 0:
                                    plate_enh = _enhance_vehicle_crop(
                                        tight, target_min_width=ocr_target_w
                                    )
                                    _ocr_and_push(
                                        plate_enh,
                                        source="veh_plate",
                                        quality=max(0.5, pconf),
                                        ocr_min=veh_ocr_min,
                                    )

                                # Crop 2: 2-line extension (always run unless
                                # we already have a long plate, e.g. ≥9 chars,
                                # which means the tight crop already captured
                                # both lines on its own).
                                if (
                                    ocr_calls < OCR_BUDGET
                                    and _best_merged_plate_len() < 9
                                    and py2_ext > py2
                                ):
                                    ext = bumper_crop[py1_pad:py2_ext, px1_pad:px2_pad]
                                    if ext.size > 0:
                                        ext_enh = _enhance_vehicle_crop(
                                            ext, target_min_width=ocr_target_w
                                        )
                                        _ocr_and_push(
                                            ext_enh,
                                            source="veh_plate_2line",
                                            quality=max(0.45, pconf * 0.9),
                                            ocr_min=veh_ocr_min,
                                        )
                                        if _best_merged_plate_len() < 9 and ocr_calls + 1 < OCR_BUDGET:
                                            _ocr_two_line_split_and_push(
                                                ext_enh,
                                                source="veh_plate_2line",
                                                quality=max(0.45, pconf * 0.9),
                                                ocr_min=veh_ocr_min,
                                            )

                                # Mark the box on the overlay. Accepted (red)
                                # if either crop produced a new plate read,
                                # otherwise candidate (orange) so the user can
                                # see "we found this region but couldn't OCR it".
                                accepted = (len(merged) > len(merged_before_tight))
                                # Best plate text we've collected so far for
                                # this region — pick the longest one which is
                                # almost certainly the most-complete read.
                                if accepted:
                                    new_plates = set(merged) - set(merged_before_tight)
                                    label = max(new_plates, key=len) if new_plates else None
                                    label_conf = merged.get(label or "", pconf) if label else pconf
                                    overlay_add_plate_region(
                                        plate_abs_bbox,
                                        text=label,
                                        conf=float(label_conf) if label_conf is not None else None,
                                        accepted=True,
                                    )
                                # No "no OCR match" box: on yellow buses the
                                # detector hits dozens of stickers/body text
                                # per frame and the user complained about
                                # "every yellow region having a box". Only
                                # validated Indian-format reads are drawn.
                    except Exception as e:
                        log.warning("  → plate YOLO on bumper crop failed: %s", e)

                # Method B: OpenCV contour-based plate finder on the bumper.
                # Run whenever we still have no accepted plate — including when
                # plate-YOLO fired on a junk box and OCR failed.
                if not merged and ocr_calls < OCR_BUDGET:
                    try:
                        from apps.vision_worker.app.plate_detection import (
                            collect_plate_region_candidates,
                        )

                        contour_cands = collect_plate_region_candidates(
                            bumper_crop, max_candidates=4, use_heuristic_bands=False,
                        )
                        log.info("  → contour fallback: %d candidates in bumper",
                                 len(contour_cands))
                        # Sort by quality and OCR the top 2 (budget permitting).
                        contour_cands.sort(
                            key=lambda d: float(d.get("quality_score") or 0.0),
                            reverse=True,
                        )
                        for cand in contour_cands[:3]:
                            if ocr_calls >= OCR_BUDGET:
                                break
                            cx1, cy1, cx2, cy2 = cand["bbox"]
                            # Map contour bbox into ABSOLUTE frame coords for the overlay.
                            abs_box = (x1 + cx1, bumper_top + cy1,
                                       x1 + cx2, bumper_top + cy2)
                            roi = cand.get("roi")
                            if roi is None or roi.size == 0:
                                continue
                            plate_rois_found = True  # at least we found regions
                            merged_before = dict(merged)
                            roi_enh = _enhance_vehicle_crop(
                                roi, target_min_width=ocr_target_w
                            )
                            _ocr_and_push(
                                roi_enh,
                                source=f"veh_cnt_{cand.get('method', '')[:6]}",
                                quality=float(cand.get("quality_score", 0.4)),
                                ocr_min=veh_ocr_min,
                            )
                            accepted = (len(merged) > len(merged_before))
                            if accepted:
                                new_p = set(merged) - set(merged_before)
                                lab = max(new_p, key=len) if new_p else None
                                lab_conf = merged.get(lab or "", 0.0) if lab else None
                                overlay_add_plate_region(
                                    abs_box, text=lab,
                                    conf=lab_conf,
                                    accepted=True,
                                )
                    except Exception as e:
                        log.warning("  → contour fallback failed: %s", e)

                # Method C: whole bumper crop if YOLO/contour still produced no read.
                if not merged and ocr_calls < OCR_BUDGET:
                    bumper_enh = _enhance_vehicle_crop(
                        bumper_crop, target_min_width=ocr_bumper_w
                    )
                    merged_before = dict(merged)
                    _ocr_and_push(
                        bumper_enh,
                        source="veh_bumper",
                        quality=0.40,
                        ocr_min=veh_ocr_min,
                    )
                    # Mark the bumper crop on the overlay so the user
                    # can see "we did try OCR on the whole bumper".
                    bumper_abs_box = (x1, bumper_top, x2, y2)
                    accepted = (len(merged) > len(merged_before))
                    if accepted:
                        new_p = set(merged) - set(merged_before)
                        lab = max(new_p, key=len) if new_p else None
                        lab_conf = merged.get(lab or "", 0.0) if lab else None
                        overlay_add_plate_region(
                            bumper_abs_box, text=lab, conf=lab_conf, accepted=True,
                        )

            if merged:
                log.info("vehicle-gated FOUND: %s (ocr_calls=%d)", list(merged.keys()), ocr_calls)
            else:
                log.info("vehicle-gated: %d vehicles scanned, no plates (ocr_calls=%d)",
                         len(vehicles), ocr_calls)
        else:
            log.debug("vehicle-gated: no vehicles detected in this frame")
    except Exception as e:
        log.warning("vehicle-gated FAILED: %s", e, exc_info=True)

    # ── SUPPLEMENT: tight plate ROIs from the standalone detector ─────────
    # Only used when no vehicles were detected (e.g. close-up gate camera).
    # Hard-capped by OCR_BUDGET so it can never blow up cycle time.
    if not merged and ocr_calls < OCR_BUDGET:
        for d in _ensure_dets():
            if ocr_calls >= OCR_BUDGET:
                break
            if float(d.get("quality_score") or 0) < q_ocr_min:
                continue
            method = str(d.get("method", ""))
            if method.startswith("heuristic_") and float(d.get("quality_score") or 0) < 0.42:
                continue
            roi = d.get("roi")
            if roi is None:
                continue
            _ocr_and_push(
                roi,
                source=method[:12] or "supp",
                quality=float(d.get("quality_score", 0.5)),
                ocr_min=ocr_floor,
            )

    # ★ Full-frame fallbacks (EasyOCR-on-frame and RapidOCR-on-frame) are
    # intentionally REMOVED. They were the source of every "SCHUOL" /
    # "ARAVALI" / "ONDUTY" misread because they OCR'd the entire bus body
    # (paint, signs, school name) as if it were a plate. If the vehicle-
    # gated path finds no plate, we'd rather post nothing than post body
    # text. Real plates can be picked up next frame — the grabber thread
    # ensures we always have a fresh one in <130 ms.

    # ── Dedupe partial reads against complete reads ──────────────────────
    # When a 2-line plate is captured both ways (tight crop yields just the
    # top line, extended crop yields the full plate), we get two entries:
    #   "HR558"     (top line, B→8 collapse)   — short, low signal
    #   "HR55BC2973" (full plate)              — long, high signal
    # The user shouldn't see two detections for the same bus. We suppress
    # any plate that is "dominated" by a longer one with the same state +
    # district prefix.
    if len(merged) > 1:
        plates_by_len = sorted(merged.keys(), key=len, reverse=True)
        suppressed: set[str] = set()
        for long in plates_by_len:
            if long in suppressed or len(long) < 7:
                continue
            long_prefix = long[:4]  # state(2) + district(≥1-2)
            for short in plates_by_len:
                if short == long or short in suppressed:
                    continue
                if len(short) >= len(long):
                    continue
                # Match on state code + at least the first district digit.
                if short[:4] == long_prefix or short[:3] == long[:3]:
                    suppressed.add(short)
                    log.info(
                        "plate dedupe: '%s' dominated by '%s' (same state+district)",
                        short, long,
                    )
        for s_drop in suppressed:
            merged.pop(s_drop, None)

    now = time.time()
    if now - _last_detection_log_ts >= 5.0:
        _last_detection_log_ts = now
        log.info(
            "sample_stack: rois=%d ocr_calls=%d merged=%s rejected=%s",
            len(dets),
            ocr_calls,
            list(merged.keys()) or "none",
            _rejected_reads[:8] or "none",
        )

    plates_out = [
        (p, merged[p]) for p in sorted(merged, key=lambda k: (-len(k), -merged[k]))
    ]
    return plates_out, overlay_drain()
