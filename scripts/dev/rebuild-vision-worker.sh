#!/usr/bin/env bash
# Rebuild and restart only the vision-worker (same compose file + profiles as start-all.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${ROOT}/infra/docker/docker-compose.yml"

compose() {
  if docker compose version &>/dev/null 2>&1; then
    docker compose -f "${COMPOSE_FILE}" "$@"
  else
    docker-compose -f "${COMPOSE_FILE}" "$@"
  fi
}

compose \
  --profile storage \
  --profile vision \
  --profile monitoring \
  build vision-worker

compose \
  --profile storage \
  --profile vision \
  --profile monitoring \
  up -d vision-worker

echo "Done. Logs: docker compose -f ${COMPOSE_FILE} logs -f vision-worker"
