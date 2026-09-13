@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   OKX 量化交易系统 - 桌面客户端
echo ============================================================
echo.

if not exist ".venv\Scripts\pythonw.exe" (
    echo [ERROR] 虚拟环境不存在
    pause
    exit /b 1
)

echo 启动中，请稍候...
:: 用 pythonw.exe 无控制台窗口启动
start "" ".venv\Scripts\pythonw.exe" desktop_app.py
echo 桌面窗口已启动！
echo.
pause
