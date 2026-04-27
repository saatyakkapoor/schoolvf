"""Vision worker configuration."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


def _detect_default_yolo_device() -> str:
    """
    Pick the best available compute device automatically.
    Honours YOLO_DEVICE env var if set (e.g. cuda:0, cpu, mps).
    Falls back to CUDA when torch reports a working GPU; else CPU.
    """
    explicit = (os.environ.get("YOLO_DEVICE") or "").strip()
    if explicit:
        return explicit
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _detect_default_ocr_gpu() -> bool:
    """EasyOCR / RapidOCR GPU defaults to ON when CUDA is reachable."""
    explicit = (os.environ.get("OCR_GPU") or "").strip().lower()
    if explicit in ("1", "true", "yes", "on"):
        return True
    if explicit in ("0", "false", "no", "off"):
        return False
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_DEFAULT_YOLO_DEVICE = _detect_default_yolo_device()
_DEFAULT_OCR_GPU = _detect_default_ocr_gpu()

# Pre-set env so any module that reads YOLO_DEVICE (singletons in this codebase)
# picks up the auto-detected GPU even when the user didn't set it explicitly.
os.environ.setdefault("YOLO_DEVICE", _DEFAULT_YOLO_DEVICE)
if _DEFAULT_YOLO_DEVICE.startswith("cuda"):
    # T1000 supports FP16 — halve memory + ~2× faster inference.
    os.environ.setdefault("YOLO_HALF", "1")


class VisionSettings(BaseSettings):
    # Default camera source (rtsp://... or webcam:N in single mode; encode @ in passwords as %40)
    CAMERA_RTSP_URL: str = (
        "rtsp://admin:abcd%401234@192.168.1.12:554/Streaming/Channels/101"
    )
    # Must match an id from GET /api/cameras (demo: cam-entry-1 / cam-exit-1).
    CAMERA_ID: str = "cam-exit-1"
    CAMERA_NAME: str = "Exit gate — Main"
    API_BASE_URL: str = "http://api:8000"
    INTERNAL_INGEST_SECRET: str = "dev-ingest-secret"
    """Must match API INTERNAL_INGEST_SECRET."""
    VISION_CAMERA_SOURCE: str = "api"
    """api: poll GET /api/internal/vision-cameras and run one camera loop per source. single: use CAMERA_ID + CAMERA_RTSP_URL (rtsp://... or webcam:N)."""
    CAMERA_POLL_INTERVAL_SEC: float = 25.0
    """How often to refresh camera list from API (URLs / active flags)."""
    PROCESS_INTERVAL_SEC: float = 0.0
    """Seconds to sleep after each processed frame. 0 = run as fast as OCR/detection allows."""
    RTSP_GRAB_DRAIN: int = 1
    """grab() this many times before retrieve(); 1 = lowest latency, higher = stabler but slower."""
    HTTP_POST_WORKERS: int = 16
    """Thread pool size for async ingest + debug HTTP (does not block the vision loop)."""
    PLATE_ENGINE: str = "live"
    """live: real RTSP + OCR. mock: demo only (no camera); requires explicit opt-in."""
    VISION_STACK: str = "sample"
    """sample: YOLO vehicle detection + EasyOCR (vehicle-first pipeline — recommended).
    rapid: RapidOCR + OpenCV ROIs (fast but misses plates on moving buses)."""
    PLATE_STAGE: str = "recognition"
    """detection: find plate regions only (logs counts, no OCR/API reads). recognition: run EasyOCR on regions."""
    PLATE_DETECT_MIN_QUALITY_FOR_OCR: float = 0.05
    """Run OCR on crops with at least this detection quality (low = try more regions, including heuristics)."""
    OCR_GPU: bool = _DEFAULT_OCR_GPU
    """EasyOCR / RapidOCR GPU. Auto-enabled when CUDA is available (e.g. T1000)."""
    YOLO_DEVICE: str = _DEFAULT_YOLO_DEVICE
    """YOLO inference device. Auto: cuda:0 if CUDA detected, else cpu. Override with YOLO_DEVICE env."""
    YOLO_HALF: bool = _DEFAULT_YOLO_DEVICE.startswith("cuda")
    """FP16 inference for YOLO on CUDA — halves VRAM, ~2× faster on T1000+. Auto-on for CUDA."""
    YOLO_IMGSZ: int = 640
    """Image size for plate-detector YOLO. 640 = best speed/accuracy on T1000."""
    PLATE_DETECTION_MODE: str = "opencv_roi"
    """Used when VISION_STACK=rapid: opencv_roi | fullframe."""
    PLATE_FILTER: str = "indian"
    """indian: HSRP-style layout only (stricter). loose: 4–12 alnum — more reads, occasional noise."""
    PLATE_ALLOWED_STATES: str = "HR,DL,CH,UP"
    """Comma-separated Indian state codes accepted by the validator. '*' = all states.
    Default 'HR,DL,CH,UP' covers the NCR (Haryana / Delhi / Chandigarh / UP) school catchment
    and rejects OCR garbage that would otherwise be 'corrected' into far-away state codes."""
    DEDUPE_SECONDS: float = 2.5
    """Min seconds between posting the same plate text again."""
    CAMERA_COOLDOWN_SEC: float = 15.0
    """After ANY plate posts from a camera, silence that camera for this many seconds.
    Prevents the same physical car being logged 3-4 times as OCR misreads it on successive frames."""
    MIN_CONFIDENCE: float = 0.09
    """OCR score floor for ROI crops. Fullframe fallback uses an even lower floor automatically.
    Lower = more reads (some noise), higher = fewer reads (misses moving-bus plates)."""
    SNAPSHOT_MAX_WIDTH: int = 520
    LIVE_DEBUG_PUSH: bool = True
    """POST structured frame summaries to API /live/debug (throttled); disable to reduce noise."""
    PLATE_USE_HEURISTIC_BANDS: bool = False
    """If True, add large center/lower frame crops as plate candidates. Often causes repeated false OCR."""
    SAMPLE_EASY_OCR_FULLFRAME_FALLBACK: bool = True
    """If True, run EasyOCR on resized full frames when ROI reads fail. More recalls; set false if too noisy."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> VisionSettings:
    s = VisionSettings()
    # Propagate to env so YOLO loaders (singletons that read os.environ) honour the choice.
    os.environ["YOLO_DEVICE"] = s.YOLO_DEVICE
    os.environ["YOLO_HALF"] = "1" if s.YOLO_HALF else "0"
    os.environ.setdefault("YOLO_IMGSZ", str(s.YOLO_IMGSZ))
    os.environ["PLATE_ALLOWED_STATES"] = s.PLATE_ALLOWED_STATES
    # Apply state-code restriction immediately so plate validator picks it up.
    try:
        from packages.shared.domain.plate import set_allowed_state_codes
        codes = (
            None if s.PLATE_ALLOWED_STATES.strip() in ("", "*")
            else [c.strip() for c in s.PLATE_ALLOWED_STATES.split(",") if c.strip()]
        )
        if codes:
            set_allowed_state_codes(codes)
    except Exception:
        pass
    # Compute pool sizing — use all 24 cores aggressively on this i7
    cpu = os.cpu_count() or 8
    os.environ.setdefault("OMP_NUM_THREADS",      str(cpu))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu))
    os.environ.setdefault("MKL_NUM_THREADS",      str(cpu))
    os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(cpu))
    os.environ.setdefault("TORCH_NUM_THREADS",    str(cpu))
    os.environ.setdefault("OPENCV_NUM_THREADS",   str(cpu))
    return s
