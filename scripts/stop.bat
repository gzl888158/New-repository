@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   OKX 量化交易系统 - 停止脚本
echo ============================================================
echo.

echo 正在停止所有 Python 进程...
taskkill /F /IM python.exe /T 2>nul

timeout /t 2 /nobreak >nul

:: 验证是否停止干净
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if %errorlevel%==0 (
    echo [WARN] 仍有 Python 进程残留
) else (
    echo [ OK ] 所有进程已停止
)

echo.
echo 系统已安全停止。
pause
exit /b 0
