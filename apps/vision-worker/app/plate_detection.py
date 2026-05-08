"""
Plate *detection* only — propose rectangular regions that look like license plates.
No OCR here. Uses several OpenCV cues (Canny contours + black-hat / morphology)
so you can tune detection before turning on recognition.

OSD exclusion: Hikvision cameras burn timestamp (top ~10%) and camera name/channel
(bottom ~10%) directly onto the video. Both zones are excluded from plate candidates
so the OCR never sees them.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("vision.plate_detection")

# ---------------------------------------------------------------------------
# Hikvision OSD burn-in zone constants (fraction of frame dimension)
# Top bar: date/time overlay (e.g. "2024-04-12 12:30:45")
# Bottom bar: camera name / channel ID (e.g. "CAM-EXIT-01")
# Both are hard-excluded so OCR never sees them.
# ---------------------------------------------------------------------------
_OSD_TOP_FRAC = 0.10    # top 10% of frame height  (Hikvision timestamp — typically top 6–8%)
_OSD_BOT_FRAC = 0.16    # bottom 16% of frame height (Hikvision camera name — typically 12–15% from bottom)

# ---------------------------------------------------------------------------
# Vehicle detection — detect bus/car/truck FIRST, then look for plates inside
# ---------------------------------------------------------------------------
# COCO class IDs for vehicles (only used when running a COCO-pretrained model)
_COCO_VEHICLE_CLASSES = {2, 3, 5, 7}  # 2=car, 3=motorcycle, 5=bus, 7=truck
_VEHICLE_CONF = 0.18  # lowered further — distant/blurry buses still need to be caught

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO = None  # type: ignore[misc, assignment]

# Path where trained models live (apps/vision-worker/models/)
_MODELS_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "models"


def _find_vehicle_model() -> str:
    """Find the best available vehicle detection model.

    Priority:
      1. YOLO_VEHICLE_MODEL env var (explicit override)
      2. vehicle_kitti_v0_best.pt in models/ (custom-trained, best for school buses)
      3. yolov8s.pt in models/ (COCO pretrained)
      4. "yolov8s.pt" (ultralytics auto-download — needs internet)
    """
    import os
    env = (os.environ.get("YOLO_VEHICLE_MODEL") or "").strip()
    if env:
        return env

    # Check local models/ directory
    for name in ("vehicle_kitti_v0_best.pt", "yolov8s.pt"):
        p = _MODELS_DIR / name
        if p.is_file():
            log.info("Found vehicle model: %s", p)
            return str(p)

    # Fallback: ultralytics will auto-download yolov8s.pt
    return "yolov8s.pt"


def _is_coco_model(model_path: str) -> bool:
    """Heuristic: if the model name contains 'yolov8' and not 'kitti'/'custom', it's COCO."""
    name = model_path.lower()
    return "yolov8" in name and "kitti" not in name and "custom" not in name


class VehicleDetector:
    """Singleton YOLO vehicle detector — finds buses/trucks/cars so we can crop and OCR them."""

    _instance: "VehicleDetector | None" = None
    _lock = __import__("threading").Lock()

    def __new__(cls) -> "VehicleDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance._loaded = False
                cls._instance._is_coco = False
                cls._instance._model_path = ""
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            # Double check in case another thread loaded it while we waited
            if self._loaded:
                return
            self._loaded = True
            if not _YOLO_AVAILABLE or _YOLO is None:
                log.warning("ultralytics YOLO not installed — vehicle detection DISABLED")
                return
            self._model_path = _find_vehicle_model()
            self._is_coco = _is_coco_model(self._model_path)
            try:
                self._model = _YOLO(self._model_path)
                self._model.fuse()
                # Move to GPU + FP16 + warm up with a dummy call so the very
                # first real frame doesn't pay the 1-2 s CUDA init cost.
                try:
                    import os as _os
                    device = (_os.environ.get("YOLO_DEVICE") or "cpu").strip()
                    half = _os.environ.get("YOLO_HALF", "0").lower() in ("1", "true", "yes")
                    try:
                        imgsz = int(_os.environ.get("YOLO_IMGSZ", "0") or "0")
                    except ValueError:
                        imgsz = 0
                    if device.startswith("cuda"):
                        self._model.to(device)
                        if half:
                            self._model.model.half()
                        warm = np.zeros((imgsz or 640, imgsz or 640, 3), dtype=np.uint8)
                        self._model(warm, conf=0.5, device=device, half=half,
                                    imgsz=imgsz or 640, verbose=False)
                        log.info(
                            "VehicleDetector warmed up on %s (half=%s imgsz=%s)",
                            device, half, imgsz or 640,
                        )
                except Exception as warm_exc:
                    log.debug("VehicleDetector warmup skipped: %s", warm_exc)
                names = getattr(self._model, "names", None)
                log.warning(
                    "VehicleDetector LOADED: %s (coco=%s) classes=%s",
                    self._model_path, self._is_coco,
                    dict(names) if names else "unknown",
                )
            except Exception as exc:
                log.warning("VehicleDetector FAILED to load %s: %s", self._model_path, exc)

    def detect(
        self,
        frame: np.ndarray,
        *,
        conf: float = _VEHICLE_CONF,
    ) -> list[dict[str, Any]]:
        """Return list of vehicle detections: {bbox, confidence, class_id, class_name}.

        Each bbox is (x1, y1, x2, y2) in pixel coords, suitable for frame[y1:y2, x1:x2] crop.
        Only returns vehicles whose centre is NOT in an OSD burn-in strip.

        For COCO models: filters to car/motorcycle/bus/truck classes.
        For custom models (KITTI etc): accepts ALL classes (they're all vehicles).
        """
        self._ensure_loaded()
        if self._model is None:
            return []

        try:
            import os
            device = (os.environ.get("YOLO_DEVICE") or "cpu").strip()
            half = os.environ.get("YOLO_HALF", "0").lower() in ("1", "true", "yes")
            try:
                imgsz = int(os.environ.get("YOLO_IMGSZ", "0") or "0")
            except ValueError:
                imgsz = 0
            yolo_kwargs = dict(conf=conf, iou=0.5, device=device, half=half, verbose=False)
            if imgsz >= 320:
                yolo_kwargs["imgsz"] = imgsz
            results = self._model(frame, **yolo_kwargs)
            detections: list[dict[str, Any]] = []
            if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
                log.debug("VehicleDetector: YOLO returned no results")
                return detections

            fh, fw = frame.shape[:2]
            boxes = results[0].boxes
            # Get model's class name mapping
            names = getattr(results[0], "names", {}) or {}

            total_boxes = len(boxes)
            skipped_class = 0
            skipped_osd = 0
            skipped_class_names: dict[str, int] = {}

            # Two-pass class filter: first try the strict COCO vehicle list.
            # If EVERY box is rejected (very common when yolov8s.pt sees a
            # head-on bus and labels it 'person' / 'truck' depending on angle
            # — happens on the user's gate camera), we fall back to accepting
            # every class on the second pass. Tiny-size and OSD filters still
            # apply, so people, traffic-light icons, and burn-in OSD text are
            # still rejected. The fallback is logged loudly so the operator
            # knows what happened.
            strict_class_filter = bool(self._is_coco)
            # Pre-scan to decide whether strict filtering would zero out the
            # frame; if so, disable it for THIS detect() call.
            if strict_class_filter:
                vehicle_class_count = 0
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id in _COCO_VEHICLE_CLASSES:
                        vehicle_class_count += 1
                if vehicle_class_count == 0 and total_boxes > 0:
                    strict_class_filter = False
                    log.warning(
                        "VehicleDetector: COCO filter would reject all %d boxes "
                        "(model classified them as non-vehicles); accepting all "
                        "classes for this frame so plate/route OCR can still run",
                        total_boxes,
                    )

            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())

                if strict_class_filter and cls_id not in _COCO_VEHICLE_CLASSES:
                    skipped_class += 1
                    cn = names.get(cls_id, f"cls{cls_id}")
                    skipped_class_names[cn] = skipped_class_names.get(cn, 0) + 1
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                c = float(box.conf[0].cpu().numpy())

                # Clamp to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)

                # Skip if vehicle centre is in OSD burn-in zone
                cy = (y1 + y2) / 2.0
                if cy < fh * _OSD_TOP_FRAC or cy > fh * (1.0 - _OSD_BOT_FRAC):
                    skipped_osd += 1
                    continue

                # Skip tiny detections (noise)
                if (x2 - x1) < 30 or (y2 - y1) < 20:
                    continue

                class_name = names.get(cls_id, f"cls{cls_id}")
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": c,
                    "class_id": cls_id,
                    "class_name": class_name,
                })

            if skipped_class_names:
                log.info(
                    "VehicleDetector: total_boxes=%d accepted=%d skipped_class=%d skipped_osd=%d non_vehicle_classes=%s",
                    total_boxes, len(detections), skipped_class, skipped_osd,
                    skipped_class_names,
                )
            else:
                log.info(
                    "VehicleDetector: total_boxes=%d accepted=%d skipped_class=%d skipped_osd=%d",
                    total_boxes, len(detections), skipped_class, skipped_osd,
                )
            return detections
        except Exception as exc:
            log.warning("VehicleDetector.detect ERROR: %s", exc, exc_info=True)
            return []


def get_vehicle_detector() -> VehicleDetector:
    """Module-level accessor for the singleton vehicle detector."""
    return VehicleDetector()


def _in_osd_zone(y1: int, y2: int, fh: int) -> bool:
    """Return True if the ROI's vertical centre falls in a Hikvision OSD burn-in strip."""
    osd_top_y = fh * _OSD_TOP_FRAC
    osd_bot_y = fh * (1.0 - _OSD_BOT_FRAC)
    cy = (y1 + y2) / 2.0
    return cy < osd_top_y or cy > osd_bot_y


def mask_osd_zones(frame: np.ndarray) -> np.ndarray:
    """
    Return a copy of *frame* with the Hikvision OSD strips blacked out.
    Use this before running fullframe OCR so the timestamp / camera-name text
    is invisible to EasyOCR / RapidOCR.
    """
    masked = frame.copy()
    fh = masked.shape[0]
    top_h = max(1, int(fh * _OSD_TOP_FRAC))
    bot_h = max(1, int(fh * _OSD_BOT_FRAC))
    masked[:top_h, :] = 0          # timestamp strip
    masked[fh - bot_h:, :] = 0    # camera-name strip
    return masked


def _is_plate_like_region(width: int, height: int, area: float) -> bool:
    if height == 0 or width < 28 or height < 10:  # lowered from 50/16 to catch distant plates
        return False
    aspect = width / height
    if not (1.8 <= aspect <= 6.5):
        return False
    bbox_area = width * height
    if bbox_area > 0 and area / bbox_area < 0.22:
        return False
    return True


def _bbox_geometry_ok(
    x1: int, y1: int, x2: int, y2: int, frame_shape: tuple[int, ...]
) -> bool:
    fh, fw = frame_shape[:2]
    if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh or x2 <= x1 or y2 <= y1:
        return False
    w, h = x2 - x1, y2 - y1
    if w < 40 or h < 14:
        return False
    if w > fw * 0.92 or h > fh * 0.5:
        return False
    ar = w / h if h else 0
    if not (1.4 <= ar <= 8.0):
        return False
    # Hard-exclude Hikvision OSD zones (timestamp top, camera name bottom)
    if _in_osd_zone(y1, y2, fh):
        return False
    return True


def _enhance_roi_for_ocr(roi: np.ndarray) -> np.ndarray | None:
    if roi is None or roi.size == 0:
        return None
    try:
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        h, w = gray.shape[:2]
        if w < 280:
            scale = 280 / max(w, 1)
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        g = cv2.bilateralFilter(g, 9, 75, 75)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return g
    except Exception:
        return None


def _roi_quality(roi: np.ndarray | None) -> float:
    if roi is None or roi.size == 0:
        return 0.0
    try:
        gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        size_score = min(1.0, (w * h) / 8000)
        contrast_score = min(1.0, float(gray.std()) / 45.0)
        edges = cv2.Canny(gray, 40, 120)
        ed = np.sum(edges > 0) / max(w * h, 1)
        edge_score = min(1.0, ed * 12)
        ar = w / h if h else 0
        aspect_score = 1.0 if 2.0 <= ar <= 5.5 else 0.45
        return size_score * 0.28 + contrast_score * 0.32 + edge_score * 0.3 + aspect_score * 0.1
    except Exception:
        return 0.0


def _candidates_from_canny(frame: np.ndarray, morph: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 180)
    edges = cv2.dilate(edges, morph, iterations=1)
    edges = cv2.erode(edges, morph, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    out: list[dict[str, Any]] = []
    fh, fw = frame.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if not _is_plate_like_region(w, h, area):
            continue
        if x + w > fw or y + h > fh:
            continue
        # Skip Hikvision OSD burn-in zones (timestamp top, camera name bottom)
        if _in_osd_zone(y, y + h, fh):
            continue
        roi_bgr = frame[y : y + h, x : x + w]
        enh = _enhance_roi_for_ocr(roi_bgr)
        if enh is None:
            continue
        q = _roi_quality(enh)
        if q < 0.12:
            continue
        bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        out.append(
            {
                "bbox": (x, y, x + w, y + h),
                "confidence": min(0.85, 0.5 + q * 0.35),
                "roi": bgr,
                "method": "canny_contour",
                "quality_score": q,
            }
        )
    return out


def _candidates_from_blackhat(frame: np.ndarray) -> list[dict[str, Any]]:
    """Dark text / plate body on brighter background — common at gates."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    out: list[dict[str, Any]] = []
    fh, fw = frame.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if not _is_plate_like_region(w, h, area):
            continue
        if x + w > fw or y + h > fh:
            continue
        # Skip Hikvision OSD burn-in zones (timestamp top, camera name bottom)
        if _in_osd_zone(y, y + h, fh):
            continue
        roi_bgr = frame[y : y + h, x : x + w]
        enh = _enhance_roi_for_ocr(roi_bgr)
        if enh is None:
            continue
        q = _roi_quality(enh)
        if q < 0.11:
            continue
        bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        out.append(
            {
                "bbox": (x, y, x + w, y + h),
                "confidence": min(0.82, 0.48 + q * 0.34),
                "roi": bgr,
                "method": "blackhat_morph",
                "quality_score": q,
            }
        )
    return out


def _iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    x1_1, y1_1, x2_1, y2_1 = a
    x1_2, y1_2, x2_2, y2_2 = b
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def merge_plate_candidates(
    items: list[dict[str, Any]],
    *,
    iou_thresh: float = 0.45,
    max_items: int = 12,
) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda d: d.get("quality_score", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for d in items:
        bb = d["bbox"]
        dup = False
        for e in kept:
            if _iou(bb, e["bbox"]) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(d)
        if len(kept) >= max_items:
            break
    return kept


def _candidates_heuristic_bands(frame: np.ndarray) -> list[dict[str, Any]]:
    """
    Large crops where people often hold a plate / printout (center + lower band).
    Printed photos often fail edge-based detectors — OCR still gets a fair shot.
    Bands start at OSD_TOP_FRAC so the timestamp region is never included.
    """
    fh, fw = frame.shape[:2]
    osd_top = int(fh * _OSD_TOP_FRAC)
    osd_bot = int(fh * (1.0 - _OSD_BOT_FRAC))
    out: list[dict[str, Any]] = []
    bands = (
        (int(fw * 0.08), max(int(fh * 0.12), osd_top), int(fw * 0.92), min(int(fh * 0.92), osd_bot), "heuristic_center"),
        (int(fw * 0.04), max(int(fh * 0.38), osd_top), int(fw * 0.96), min(int(fh * 0.96), osd_bot), "heuristic_lower"),
        (int(fw * 0.18), max(int(fh * 0.22), osd_top), int(fw * 0.82), min(int(fh * 0.78), osd_bot), "heuristic_tight"),
    )
    for x1, y1, x2, y2, method in bands:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 + 80 or y2 <= y1 + 40:
            continue
        roi_bgr = frame[y1:y2, x1:x2]
        enh = _enhance_roi_for_ocr(roi_bgr)
        if enh is None:
            continue
        bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        q = max(0.18, _roi_quality(enh) * 0.85)
        out.append(
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.55,
                "roi": bgr,
                "method": method,
                "quality_score": q,
            }
        )
    return out


def collect_plate_region_candidates(
    frame: np.ndarray,
    *,
    max_candidates: int = 16,
    use_heuristic_bands: bool = False,
) -> list[dict[str, Any]]:
    """
    Return plate-shaped region proposals: each dict has bbox, roi (BGR for OCR), quality_score, method.

    Heuristic bands are large crops (center / lower frame). They are off by default because EasyOCR
    often returns the same plausible-looking plate text on unrelated video.

    OSD zones (top/bottom 10%) are excluded from all candidates so the Hikvision timestamp and
    camera-name burn-in text never reaches the OCR engine.
    """
    morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fh, fw = frame.shape[:2]
    raw: list[dict[str, Any]] = []
    if use_heuristic_bands:
        raw.extend(_candidates_heuristic_bands(frame))
    raw.extend(_candidates_from_canny(frame, morph))
    raw.extend(_candidates_from_blackhat(frame))

    merged = merge_plate_candidates(raw, iou_thresh=0.42, max_items=max_candidates * 2)
    validated: list[dict[str, Any]] = []
    for d in merged:
        x1, y1, x2, y2 = d["bbox"]
        if not _bbox_geometry_ok(x1, y1, x2, y2, frame.shape):
            continue
        if d.get("quality_score", 0) < 0.08:
            continue
        validated.append(d)
    return validated[:max_candidates]
