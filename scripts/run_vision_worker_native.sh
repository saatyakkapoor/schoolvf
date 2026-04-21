#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the vision worker NATIVELY on this Mac/Linux machine.
#
# Why native?  Docker on Mac is a Linux VM — it cannot access USB or built-in
# cameras. Running natively gives OpenCV direct access via AVFoundation (Mac)
# or V4L2 (Linux).
#
# Prerequisites (once):
#   pip install rapidocr-onnxruntime opencv-python httpx pydantic-settings
#
# Usage:
#   chmod +x scripts/run_vision_worker_native.sh
#   ./scripts/run_vision_worker_native.sh
#
# Stop the Docker vision-worker container first to avoid duplicate processing:
#   docker compose -f infra/docker/docker-compose.yml stop vision-worker
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# ── environment ────────────────────────────────────────────────────────────
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
export INTERNAL_INGEST_SECRET="${INTERNAL_INGEST_SECRET:-dev-ingest-secret}"

export VISION_CAMERA_SOURCE="${VISION_CAMERA_SOURCE:-api}"
export CAMERA_POLL_INTERVAL_SEC="${CAMERA_POLL_INTERVAL_SEC:-25}"

export PLATE_ENGINE="${PLATE_ENGINE:-rapidocr}"
export VISION_STACK="${VISION_STACK:-rapid}"
export PLATE_FILTER="${PLATE_FILTER:-indian}"
export PLATE_STAGE="${PLATE_STAGE:-recognition}"
export PLATE_DETECTION_MODE="${PLATE_DETECTION_MODE:-opencv_roi}"
export MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.16}"

export DEDUPE_SECONDS="${DEDUPE_SECONDS:-4}"
export SNAPSHOT_MAX_WIDTH="${SNAPSHOT_MAX_WIDTH:-640}"
export PROCESS_INTERVAL_SEC="${PROCESS_INTERVAL_SEC:-0}"
export RTSP_GRAB_DRAIN="${RTSP_GRAB_DRAIN:-1}"
export HTTP_POST_WORKERS="${HTTP_POST_WORKERS:-4}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "═══════════════════════════════════════════════════════════"
echo "  Vision worker — NATIVE mode (cameras accessible)"
echo "  API_BASE_URL : $API_BASE_URL"
echo "  Camera source: $VISION_CAMERA_SOURCE (polls API every ${CAMERA_POLL_INTERVAL_SEC}s)"
echo ""
echo "  ⚠  Stop the Docker vision-worker first:"
echo "     docker compose -f infra/docker/docker-compose.yml stop vision-worker"
echo "═══════════════════════════════════════════════════════════"

exec python -m apps.vision_worker.app.main
