@echo off
chcp 65001 >nul
title OKX Radical Trading System

echo ========================================
echo  OKX 激进型量化交易系统启动
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] 启动主交易系统...
start "OKX Trading System" cmd /k ".venv\Scripts\python.exe main.py"

timeout /t 5 /nobreak >nul

echo [2/2] 启动可视化控制面板...
start "OKX Dashboard" cmd /k ".venv\Scripts\python.exe dashboard_api.py"

echo.
echo ========================================
echo  系统启动完成！
echo  可视化面板: http://localhost:8080
echo ========================================
echo.
pause
