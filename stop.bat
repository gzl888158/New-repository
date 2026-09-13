@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 虚拟环境不存在
    pause
    exit /b 1
)

".venv\Scripts\python.exe" stop.py
pause
