@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual env not found. Run: python -m venv .venv
    pause
    exit /b 1
)

".venv\Scripts\python.exe" start.py
if %errorlevel% neq 0 pause
