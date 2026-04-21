"""Vision worker configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings


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
    OCR_GPU: bool = False
    """EasyOCR gpu=… (sample AdaptiveOCRProcessor)."""
    PLATE_DETECTION_MODE: str = "opencv_roi"
    """Used when VISION_STACK=rapid: opencv_roi | fullframe."""
    PLATE_FILTER: str = "indian"
    """indian: HSRP-style layout only (stricter). loose: 4–12 alnum — more reads, occasional noise."""
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
    return VisionSettings()
