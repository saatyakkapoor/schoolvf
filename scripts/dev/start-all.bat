@echo off
REM One-click launcher for the SchoolVF stack on Windows.
REM Forwards all CLI args to start-all.ps1 (e.g. -HostIp 192.168.1.50 -NoBuild).
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-all.ps1" %*
exit /b %ERRORLEVEL%
