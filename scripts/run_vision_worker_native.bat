@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM Run the vision worker NATIVELY on Windows (DirectShow camera access).
REM
REM Why native?  Docker on Windows cannot access USB cameras by default.
REM Running natively gives OpenCV direct access via DirectShow.
REM
REM Prerequisites (once):
REM   pip install rapidocr-onnxruntime opencv-python httpx pydantic-settings
REM
REM Stop the Docker vision-worker first:
REM   docker compose -f infra/docker/docker-compose.yml stop vision-worker
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0\.."

IF NOT DEFINED API_BASE_URL         SET API_BASE_URL=http://localhost:8000
IF NOT DEFINED INTERNAL_INGEST_SECRET SET INTERNAL_INGEST_SECRET=dev-ingest-secret
IF NOT DEFINED VISION_CAMERA_SOURCE SET VISION_CAMERA_SOURCE=api
IF NOT DEFINED CAMERA_POLL_INTERVAL_SEC SET CAMERA_POLL_INTERVAL_SEC=25
IF NOT DEFINED PLATE_ENGINE         SET PLATE_ENGINE=rapidocr
IF NOT DEFINED VISION_STACK         SET VISION_STACK=rapid
IF NOT DEFINED PLATE_FILTER         SET PLATE_FILTER=indian
IF NOT DEFINED PLATE_STAGE          SET PLATE_STAGE=recognition
IF NOT DEFINED PLATE_DETECTION_MODE SET PLATE_DETECTION_MODE=opencv_roi
IF NOT DEFINED MIN_CONFIDENCE       SET MIN_CONFIDENCE=0.16
IF NOT DEFINED DEDUPE_SECONDS       SET DEDUPE_SECONDS=4
IF NOT DEFINED SNAPSHOT_MAX_WIDTH   SET SNAPSHOT_MAX_WIDTH=640
IF NOT DEFINED PROCESS_INTERVAL_SEC SET PROCESS_INTERVAL_SEC=0
IF NOT DEFINED HTTP_POST_WORKERS    SET HTTP_POST_WORKERS=4

SET PYTHONPATH=%CD%;%PYTHONPATH%

echo ═══════════════════════════════════════════════════
echo   Vision worker — NATIVE mode (cameras accessible)
echo   API_BASE_URL : %API_BASE_URL%
echo   Camera source: %VISION_CAMERA_SOURCE%
echo ═══════════════════════════════════════════════════

python -m apps.vision_worker.app.main
