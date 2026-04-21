@echo off
title SchoolVF — Start
echo Starting SchoolVF services...
cd /d "C:\SchoolVF"
docker compose -f infra\docker\docker-compose.yml up -d
echo.
echo Done. Opening dashboard...
start http://localhost:3000
timeout /t 2 /nobreak >nul
