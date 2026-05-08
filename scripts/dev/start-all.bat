@echo off
REM One-click launcher for the SchoolVF stack on Windows.
REM Forwards all CLI args to start-all.ps1 (e.g. -HostIp 192.168.1.50 -NoBuild).
setlocal
set "SCRIPT_DIR=%~dp0"
echo [start-all.bat] launching PowerShell host: "%SCRIPT_DIR%start-all.ps1"
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-all.ps1" %*
set "EC=%ERRORLEVEL%"
echo.
echo [start-all.bat] PowerShell exited with code %EC%
echo.
echo Press any key to close this window...
pause >nul
exit /b %EC%
