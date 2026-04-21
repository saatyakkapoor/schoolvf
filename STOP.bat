@echo off
title SchoolVF — Stop
echo Stopping SchoolVF services...
cd /d "C:\SchoolVF"
docker compose -f infra\docker\docker-compose.yml down
echo.
echo All services stopped.
timeout /t 2 /nobreak >nul
