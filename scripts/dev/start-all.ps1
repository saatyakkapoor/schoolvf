<#
.SYNOPSIS
    Start the full SchoolVF stack on Windows (PowerShell native).

.DESCRIPTION
    Mirrors scripts/dev/start-all.sh:
      - Builds & starts every service via docker compose with
        storage + vision + monitoring profiles enabled.
      - Auto-detects the primary LAN IPv4 so URLs are reachable
        from other devices on the network. Override with -HostIp.

.PARAMETER HostIp
    Override the auto-detected LAN IP (useful on multi-NIC machines).

.PARAMETER NoBuild
    Skip the image build step (faster restarts when nothing changed).

.PARAMETER ExtraArgs
    Extra arguments forwarded to `docker compose` (e.g. --project-name).

.EXAMPLE
    .\scripts\dev\start-all.ps1
    .\scripts\dev\start-all.ps1 -HostIp 192.168.1.50
    .\scripts\dev\start-all.ps1 -NoBuild
#>
[CmdletBinding()]
param(
    [string]$HostIp,
    [switch]$NoBuild,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root        = Resolve-Path (Join-Path $ScriptDir '..\..')
$ComposeDir  = Join-Path $Root 'infra\docker'
$ComposeFile = Join-Path $ComposeDir 'docker-compose.yml'

if (-not (Test-Path $ComposeFile)) {
    Write-Error "docker-compose.yml not found at $ComposeFile"
    exit 1
}

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ComposeArgs)

    # Prefer `docker compose` (v2 plugin); fall back to legacy `docker-compose`.
    & docker compose version *> $null
    if ($LASTEXITCODE -eq 0) {
        & docker compose -f $ComposeFile @ComposeArgs
    } else {
        & docker-compose -f $ComposeFile @ComposeArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose exited with code $LASTEXITCODE"
    }
}

# --- Sanity: Docker daemon reachable? ---------------------------------
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker does not appear to be running. Start Docker Desktop and retry."
    exit 1
}

# --- Detect the host LAN IP ------------------------------------------
function Get-PrimaryLanIp {
    if ($env:HOST_IP) { return $env:HOST_IP }

    try {
        # Use the route table to find the interface used for outbound
        # traffic (works whether you're on Wi-Fi or Ethernet).
        $route = Find-NetRoute -RemoteIPAddress 8.8.8.8 -ErrorAction Stop |
                 Select-Object -First 1
        if ($route -and $route.IPAddress) { return $route.IPAddress }
    } catch { }

    # Fallback: first non-loopback, non-link-local IPv4 with a default
    # gateway, ranked Up > rest.
    try {
        $candidate = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
            Where-Object {
                $_.IPAddress -notlike '127.*' -and
                $_.IPAddress -notlike '169.254.*' -and
                $_.PrefixOrigin -in 'Dhcp','Manual'
            } |
            Sort-Object -Property @{Expression='InterfaceMetric';Descending=$false} |
            Select-Object -First 1 -ExpandProperty IPAddress
        if ($candidate) { return $candidate }
    } catch { }

    return 'localhost'
}

if (-not $HostIp) { $HostIp = Get-PrimaryLanIp }

# --- Bring the stack up ---------------------------------------------
Write-Host "Starting full stack from $ComposeFile (storage + vision + monitoring profiles)..." -ForegroundColor Cyan

$composeArgs = @(
    '--profile','storage',
    '--profile','vision',
    '--profile','monitoring'
)
if ($ExtraArgs) { $composeArgs += $ExtraArgs }
$composeArgs += 'up','-d'
if (-not $NoBuild) { $composeArgs += '--build' }

Invoke-Compose @composeArgs

# --- Print LAN URLs --------------------------------------------------
$dashPort  = if ($env:DASHBOARD_PORT)  { $env:DASHBOARD_PORT }  else { '3000' }
$apiPort   = if ($env:API_PORT)        { $env:API_PORT }        else { '8000' }
$redisPort = if ($env:REDIS_PORT)      { $env:REDIS_PORT }      else { '6379' }
$grafPort  = if ($env:GRAFANA_PORT)    { $env:GRAFANA_PORT }    else { '3001' }
$promPort  = if ($env:PROMETHEUS_PORT) { $env:PROMETHEUS_PORT } else { '9090' }
$minioApi  = if ($env:MINIO_API_PORT)  { $env:MINIO_API_PORT }  else { '9000' }
$minioUi   = if ($env:MINIO_CONSOLE_PORT) { $env:MINIO_CONSOLE_PORT } else { '9001' }

Write-Host ""
Write-Host "Stack is reachable on the LAN at $HostIp :" -ForegroundColor Green
Write-Host "  Dashboard:   http://${HostIp}:${dashPort}"
Write-Host "  API:         http://${HostIp}:${apiPort}"
Write-Host "  Redis:       ${HostIp}:${redisPort}"
Write-Host "  Grafana:     http://${HostIp}:${grafPort}"
Write-Host "  Prometheus:  http://${HostIp}:${promPort}"
Write-Host "  MinIO API:   http://${HostIp}:${minioApi}"
Write-Host "  MinIO UI:    http://${HostIp}:${minioUi}"
Write-Host ""

if ($HostIp -eq 'localhost') {
    Write-Host "Could not auto-detect a LAN IP. Run with -HostIp 192.168.x.y if needed." -ForegroundColor Yellow
} else {
    Write-Host "Open the dashboard from any device on the same network using the URL above."
    Write-Host "If Windows Firewall blocks inbound 3000/8000, allow Docker Desktop in Windows Defender Firewall." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "Vision worker: schoolvf-vision-worker (logs: docker compose -f $ComposeFile logs -f vision-worker)"
Write-Host "All logs:      docker compose -f $ComposeFile logs -f"
