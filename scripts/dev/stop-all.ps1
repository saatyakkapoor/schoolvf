<#
.SYNOPSIS
    Stop the full SchoolVF docker stack on Windows.

.PARAMETER PurgeVolumes
    Also remove volumes (wipes Postgres + MinIO data). USE WITH CARE.

.EXAMPLE
    .\scripts\dev\stop-all.ps1
    .\scripts\dev\stop-all.ps1 -PurgeVolumes
#>
[CmdletBinding()]
param(
    [switch]$PurgeVolumes
)

$ErrorActionPreference = 'Stop'
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeFile = Resolve-Path (Join-Path $ScriptDir '..\..\infra\docker\docker-compose.yml')

$composeArgs = @(
    '-f', $ComposeFile,
    '--profile','storage',
    '--profile','vision',
    '--profile','monitoring',
    'down'
)
if ($PurgeVolumes) { $composeArgs += '-v' }

& docker compose version *> $null
if ($LASTEXITCODE -eq 0) {
    & docker compose @composeArgs
} else {
    & docker-compose @composeArgs
}
