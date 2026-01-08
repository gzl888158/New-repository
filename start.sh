#!/bin/bash
echo "=============== 欧易网格交易机器人 一键启动 ==============="
# 安装依赖
echo "1. 安装依赖包..."
pip3 install -r requirements.txt

# 启动后端服务
echo "2. 启动后端API服务..."
nohup python3 -m src.main > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "后端服务PID：$BACKEND_PID"

# 启动前端页面
echo "3. 启动前端页面..."
if command -v xdg-open &> /dev/null; then
    xdg-open index.html
elif command -v open &> /dev/null; then
    open index.html
else
    echo "请手动打开 index.html 文件"
fi

echo "=============== 启动完成！ ==============="
echo "后端地址：http://localhost:8000"
echo "日志目录：./logs"
echo "停止脚本：./stop.sh"