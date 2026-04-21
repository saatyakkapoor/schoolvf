#!/usr/bin/env bash
# Full stack by default: Postgres, Redis, API, dashboard, MinIO, vision-worker,
# Prometheus, and Grafana (--profile storage + vision + monitoring).
# Redis has no profile in compose; it always starts with the core services.
#
# To start a slimmer set, use compose directly with fewer --profile flags.
#
# Examples:
#   ./scripts/dev/start-all.sh
# Extra args are compose globals (e.g. --project-name), not service names.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_DIR="${ROOT}/infra/docker"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"

compose() {
  if docker compose version &>/dev/null 2>&1; then
    docker compose -f "${COMPOSE_FILE}" "$@"
  else
    docker-compose -f "${COMPOSE_FILE}" "$@"
  fi
}

if ! docker info &>/dev/null; then
  echo "Docker does not appear to be running. Start Docker Desktop (or the docker daemon) and retry." >&2
  exit 1
fi

echo "Starting full stack from ${COMPOSE_FILE} (storage + vision + monitoring profiles)..."
# --profile must come *before* the subcommand (e.g. up); after `up` it is rejected.
compose \
  --profile storage \
  --profile vision \
  --profile monitoring \
  "$@" \
  up -d --build

echo ""
echo "URLs (defaults from docker-compose):"
echo "  Dashboard:   http://localhost:${DASHBOARD_PORT:-3000}"
echo "  API:         http://localhost:${API_PORT:-8000}"
echo "  Redis:       localhost:${REDIS_PORT:-6379}"
echo "  Grafana:     http://localhost:${GRAFANA_PORT:-3001}"
echo "  Prometheus:  http://localhost:${PROMETHEUS_PORT:-9090}"
echo "  MinIO API:   http://localhost:${MINIO_API_PORT:-9000}"
echo "  MinIO UI:    http://localhost:${MINIO_CONSOLE_PORT:-9001}"
echo ""
echo "Vision worker: schoolvf-vision-worker (see: compose logs vision-worker)"
echo ""
echo "Logs: docker compose -f ${COMPOSE_FILE} logs -f"
