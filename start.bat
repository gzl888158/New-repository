@echo off
echo =============== 欧易网格交易机器人 一键启动 ===============
echo 1. 安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 依赖安装失败，请检查Python和pip是否配置正确
    pause
    exit /b 1
)

echo 2. 启动后端API服务...
start cmd /k "python -m src.main"
echo 后端服务已启动，端口：8000

echo 3. 打开前端页面...
start index.html

echo =============== 启动完成！ ===============
echo 后端地址：http://localhost:8000
echo 日志目录：./logs
echo 关闭后端服务请关闭对应的cmd窗口
pause