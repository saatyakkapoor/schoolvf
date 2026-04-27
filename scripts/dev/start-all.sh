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

# Detect the primary LAN IP so URLs are reachable from other devices on the
# network. Honour HOST_IP if the operator has already exported one. Falls back
# to "localhost" so the script never breaks on a disconnected machine.
detect_host_ip() {
  if [[ -n "${HOST_IP:-}" ]]; then
    printf '%s' "${HOST_IP}"
    return
  fi
  local ip=""
  case "$(uname -s)" in
    Darwin)
      # Try the default-route interface first, then en0/en1, then any
      # other interface that has an IPv4 (skip loopback / VPN tunnels /
      # link-local 169.254.* addresses).
      local iface
      iface=$(route -n get default 2>/dev/null | awk '/interface:/ {print $2; exit}')
      if [[ -n "${iface}" && "${iface}" != utun* ]]; then
        ip=$(ipconfig getifaddr "${iface}" 2>/dev/null || true)
      fi
      [[ -z "${ip}" ]] && ip=$(ipconfig getifaddr en0 2>/dev/null || true)
      [[ -z "${ip}" ]] && ip=$(ipconfig getifaddr en1 2>/dev/null || true)
      if [[ -z "${ip}" ]]; then
        ip=$(ifconfig 2>/dev/null \
          | awk '/^[a-z]/ {iface=$1} /inet / && $2 !~ /^127\./ && $2 !~ /^169\.254\./ && iface !~ /^(lo|utun|awdl|llw|bridge)/ {print $2; exit}')
      fi
      ;;
    Linux)
      ip=$(hostname -I 2>/dev/null | awk '{print $1}')
      [[ -z "${ip}" ]] && ip=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1)
      ;;
    MINGW*|MSYS*|CYGWIN*)
      ip=$(powershell.exe -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 -PrefixOrigin Dhcp,Manual | Where-Object { \$_.IPAddress -notlike '169.*' -and \$_.IPAddress -ne '127.0.0.1' } | Select-Object -First 1 -ExpandProperty IPAddress)" 2>/dev/null | tr -d '\r')
      ;;
  esac
  printf '%s' "${ip:-localhost}"
}

HOST_IP=$(detect_host_ip)

echo ""
echo "Stack is reachable on the LAN at ${HOST_IP}:"
echo "  Dashboard:   http://${HOST_IP}:${DASHBOARD_PORT:-3000}"
echo "  API:         http://${HOST_IP}:${API_PORT:-8000}"
echo "  Redis:       ${HOST_IP}:${REDIS_PORT:-6379}"
echo "  Grafana:     http://${HOST_IP}:${GRAFANA_PORT:-3001}"
echo "  Prometheus:  http://${HOST_IP}:${PROMETHEUS_PORT:-9090}"
echo "  MinIO API:   http://${HOST_IP}:${MINIO_API_PORT:-9000}"
echo "  MinIO UI:    http://${HOST_IP}:${MINIO_CONSOLE_PORT:-9001}"
echo ""
if [[ "${HOST_IP}" == "localhost" ]]; then
  echo "Could not auto-detect a LAN IP. Export HOST_IP=192.168.x.y and rerun if needed."
else
  echo "Open the dashboard from any device on the same network using the URL above."
fi
echo ""
echo "Vision worker: schoolvf-vision-worker (see: compose logs vision-worker)"
echo "Logs: docker compose -f ${COMPOSE_FILE} logs -f"
