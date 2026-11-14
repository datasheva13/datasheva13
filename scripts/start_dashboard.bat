@echo off
rem Double-click to open the review dashboard in your browser.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m moto_shorts dashboard
