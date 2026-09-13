@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   OKX 量化交易系统 - 统一启动脚本
echo ============================================================
echo.

:: 检查 Python 环境
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] 虚拟环境不存在，请先运行: python -m venv .venv
    pause
    exit /b 1
)

:: 仅关闭本项目相关的 Python 进程（watchdog.py / main.py / dashboard_api.py）
echo [1/3] 清理已有进程...
for /f "tokens=2" %%p in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /C:"watchdog.py" /C:"main.py" /C:"dashboard_api.py" 2^>nul') do (
    set "pid=%%~p"
    if defined pid taskkill /F /PID %%~p /T 2>nul
)
echo       清理完成

:: 创建日志目录
if not exist "logs" mkdir logs
if not exist "data" mkdir data

:: 启动 watchdog（负责监控和重启 main.py）
echo [2/3] 启动 Watchdog + 交易主程序...
start "OKX-Watchdog" .venv\Scripts\python.exe watchdog.py
:: 获取刚启动的进程 PID
timeout /t 1 /nobreak >nul
for /f "tokens=2" %%p in ('tasklist /FI "WINDOWTITLE eq OKX-Watchdog*" /FO CSV 2^>nul') do set "WPID=%%~p"
if defined WPID (echo       Watchdog PID: %WPID%) else (echo       Watchdog PID: 已启动（PID无法通过窗口标题获取）)

:: 等待 main.py 初始化
timeout /t 3 /nobreak >nul

:: 启动 Dashboard API 服务器
echo [3/3] 启动 Dashboard 服务器...
start "OKX-Dashboard" .venv\Scripts\python.exe dashboard_api.py
timeout /t 1 /nobreak >nul
for /f "tokens=2" %%p in ('tasklist /FI "WINDOWTITLE eq OKX-Dashboard*" /FO CSV 2^>nul') do set "DPID=%%~p"
if defined DPID (echo       Dashboard PID: %DPID%) else (echo       Dashboard PID: 已启动)

echo.
echo ============================================================
echo   启动完成！
echo.
echo   Dashboard:  http://localhost:8080
echo   交易主程序: 由 Watchdog 管理（崩溃自动重启）
echo   DASHBOARD_TOKEN: 已配置（见 .env）
echo ============================================================
echo.
echo   关闭方式: 双击 stop.bat 或关闭命令行窗口
echo.

timeout /t 5 /nobreak >nul
exit /b 0
