#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "Starting native vision worker for local webcams (Mac/Windows)."
echo "Stopping docker vision-worker service so it does not spam retry logs..."
docker compose -f "$ROOT_DIR/infra/docker/docker-compose.yml" stop vision-worker >/dev/null || true

if [[ -z "${INTERNAL_INGEST_SECRET:-}" ]]; then
  export INTERNAL_INGEST_SECRET="dev-ingest-secret"
fi
if [[ -z "${API_BASE_URL:-}" ]]; then
  export API_BASE_URL="http://localhost:8000"
fi
if [[ -z "${VISION_CAMERA_SOURCE:-}" ]]; then
  export VISION_CAMERA_SOURCE="api"
fi

if [[ -z "${CAMERA_RTSP_URL:-}" ]]; then
  export CAMERA_RTSP_URL="webcam:0"
fi

echo
echo "Using:"
echo "  API_BASE_URL=$API_BASE_URL"
echo "  VISION_CAMERA_SOURCE=$VISION_CAMERA_SOURCE"
echo "  CAMERA_RTSP_URL=$CAMERA_RTSP_URL"
echo "  INTERNAL_INGEST_SECRET=$INTERNAL_INGEST_SECRET"
echo
echo "Tip: list camera indices with:"
echo "  python3 scripts/list_opencv_cameras.py --max 6"
echo
if ! python3 - <<'PY'
import cv2  # noqa: F401
import pydantic_settings  # noqa: F401
import httpx  # noqa: F401
PY
then
  echo "Missing Python deps for native vision worker."
  echo "Install once with:"
  echo "  python3 -m pip install -r apps/vision-worker/requirements.txt"
  exit 1
fi

echo "Launching worker..."
exec python3 -m apps.vision_worker.app.main
