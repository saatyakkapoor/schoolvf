@echo off
setlocal EnableDelayedExpansion
title SchoolVF — Windows Setup

echo.
echo =====================================================================
echo SchoolVF ^| School Bus Entry-Exit Monitoring System
echo The Shri Ram School Aravali
echo Windows Auto-Setup Script
echo =====================================================================
echo.

:: ── 0. Must run as Administrator ─────────────────────────────────────────
net session >nul 2>&1
if %errorlevel% neq 0 (
echo [ERROR] Please right-click this file and choose "Run as administrator".
pause
exit /b 1
)

:: ── 1. Check Docker ───────────────────────────────────────────────────────
echo [1/5] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
echo.
echo [ERROR] Docker Desktop is not installed or not in PATH.
echo Install it from https://www.docker.com/products/docker-desktop/
echo then run this script again.
pause
exit /b 1
)
docker info >nul 2>&1
if %errorlevel% neq 0 (
echo.
echo [ERROR] Docker is not running.
echo Start Docker Desktop from the Start Menu, wait for the whale icon
echo in the system tray to stop animating, then run this script again.
pause
exit /b 1
)
echo [OK] Docker is running.

:: ── 2. Copy project to C:\SchoolVF ───────────────────────────────────────
echo.
echo [2/5] Installing project to C:\SchoolVF...
set "DEST=C:\SchoolVF"
set "SRC=%~dp0"

if not exist "%DEST%" mkdir "%DEST%"
robocopy "%SRC%" "%DEST%" /MIR /XD ".git" "node_modules" "__pycache__" ".venv" /XF "*.pyc" "SETUP_WINDOWS.bat" /NFL /NDL /NP /NJH >nul
echo [OK] Project copied to C:\SchoolVF
cd /d "%DEST%"

:: ── 3. Install Python vision worker deps ─────────────────────────────────
echo.
echo [3/5] Installing Python vision worker dependencies...
echo (Downloads ~500MB — may take 5-10 minutes on first run)
echo.
python -m pip install --upgrade pip --quiet
python -m pip install rapidocr-onnxruntime opencv-python httpx pydantic-settings numpy --quiet
if %errorlevel% neq 0 (
echo [WARN] Some Python packages failed. USB webcams may not work.
echo RTSP cameras will still work fine via Docker.
) else (
echo [OK] Python dependencies installed.
)

:: ── 4. Build ALL containers (including vision worker) ────────────────────
echo.
echo [4/5] Building all application containers...
echo (First build takes 5-15 minutes — please wait)
echo.
docker compose -f infra\docker\docker-compose.yml --profile vision build
if %errorlevel% neq 0 (
echo.
echo [ERROR] Docker build failed. See output above.
pause
exit /b 1
)
echo [OK] All containers built.

:: ── 5. Start services ────────────────────────────────────────────────────
echo.
echo [5/5] Starting SchoolVF services...
docker compose -f infra\docker\docker-compose.yml --profile vision up -d
if %errorlevel% neq 0 (
echo.
echo [ERROR] Failed to start services.
echo Run: docker compose -f infra\docker\docker-compose.yml logs
pause
exit /b 1
)

:: Wait for API
echo.
echo Waiting for API to be ready...
set /a tries=0
:HEALTHCHECK
set /a tries+=1
if %tries% gtr 40 goto HEALTHY_TIMEOUT
timeout /t 3 /nobreak >nul
curl -fsS http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 goto HEALTHY
echo Waiting... (%tries%/40)
goto HEALTHCHECK

:HEALTHY_TIMEOUT
echo [WARN] API did not respond in time — it may still be starting.
goto SHORTCUTS

:HEALTHY
echo [OK] API is healthy.

:: ── Desktop shortcuts ─────────────────────────────────────────────────────
:SHORTCUTS
echo.
echo Creating desktop shortcuts...
powershell -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\SchoolVF Dashboard.url'); $s.TargetPath='http://localhost:3000'; $s.Save()"
powershell -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Start SchoolVF.lnk'); $s.TargetPath='C:\SchoolVF\START.bat'; $s.WorkingDirectory='C:\SchoolVF'; $s.Save()"
powershell -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Stop SchoolVF.lnk'); $s.TargetPath='C:\SchoolVF\STOP.bat'; $s.WorkingDirectory='C:\SchoolVF'; $s.Save()"
powershell -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\SchoolVF Vision Worker (USB Camera).lnk'); $s.TargetPath='C:\SchoolVF\scripts\run_vision_worker_native.bat'; $s.WorkingDirectory='C:\SchoolVF'; $s.Save()"
echo [OK] Desktop shortcuts created.

start http://localhost:3000

echo.
echo =====================================================================
echo SETUP COMPLETE
echo =====================================================================
echo.
echo Dashboard : http://localhost:3000
echo API : http://localhost:8000
echo.
echo Desktop shortcuts on your desktop:
echo "SchoolVF Dashboard" — opens the dashboard
echo "Start SchoolVF" — start all services
echo "Stop SchoolVF" — stop all services
echo "SchoolVF Vision Worker (USB Camera)" — native worker for webcams
echo.
echo =====================================================================
echo.
pause