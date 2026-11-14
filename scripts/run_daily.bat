@echo off
rem Called by Task Scheduler. Logs go to work\logs\daily_YYYY-MM-DD.log
cd /d "%~dp0.."
if not exist work\logs mkdir work\logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set D=%%i
".venv\Scripts\python.exe" -m moto_shorts daily >> "work\logs\daily_%D%.log" 2>&1
