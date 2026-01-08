@echo off
title 欧易网格机器人 一键自检工具
echo ==============================================
echo 欧易网格机器人 本地环境自检工具（Windows版）
echo ==============================================
echo 检测时间：%date% %time%
echo.

:: 定义核心文件/目录列表
set "required_dirs=config src logs"
set "required_config=config\config.yaml"
set "required_src=src\__init__.py src\main.py src\okx_api.py src\utils.py"
set "required_root=start.bat stop.bat start.sh stop.sh .env index.html requirements.txt"
set "port=8000"

:: 1. 检查核心目录是否存在
echo 【1】核心目录检查
for %%d in (%required_dirs%) do (
    if exist %%d (
        echo [✅] 目录 %%d 存在
    ) else (
        echo [❌] 目录 %%d 缺失！请创建
        set "error=1"
    )
)
echo.

:: 2. 检查核心文件是否存在
echo 【2】核心文件检查
:: config目录
if exist %required_config% (
    echo [✅] config\config.yaml 存在
) else (
    echo [❌] config\config.yaml 缺失！请创建并粘贴配置
    set "error=1"
)
:: src目录
for %%f in (%required_src%) do (
    if exist %%f (
        echo [✅] %%f 存在
    ) else (
        echo [❌] %%f 缺失！请创建并粘贴代码
        set "error=1"
    )
)
:: 根目录
for %%f in (%required_root%) do (
    if exist %%f (
        echo [✅] %%f 存在
    ) else (
        echo [⚠️] %%f 缺失（Windows可忽略sh脚本，Linux/Mac可忽略bat脚本）
    )
)
echo.

:: 3. 检查端口是否被占用
echo 【3】后端端口（%port%）占用检查
netstat -ano | findstr :%port% >nul
if %errorlevel% equ 0 (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%port%') do (
        echo [⚠️] 端口 %port% 被进程PID %%a 占用！请结束进程后重启
    )
) else (
    echo [✅] 端口 %port% 未被占用
)
echo.

:: 4. 检查后端服务是否运行
echo 【4】后端服务运行状态检查
tasklist | findstr /i "python.exe" | findstr /i "src.main" >nul
if %errorlevel% equ 0 (
    echo [✅] 后端服务正在运行
) else (
    echo [❌] 后端服务未运行！请执行 start.bat 启动
    set "error=1"
)
echo.

:: 5. 自检结果汇总
echo ==============================================
if defined error (
    echo [❌] 自检未通过！请修复上述问题后重试
) else (
    echo [✅] 自检通过！请本地打开 index.html 开始测试
)
pause