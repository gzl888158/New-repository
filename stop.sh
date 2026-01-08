#!/bin/bash
echo "=============== 欧易网格交易机器人 一键停止 ==============="
# 停止后端服务
BACKEND_PID=$(ps aux | grep "src.main" | grep -v grep | awk '{print $2}')
if [ -n "$BACKEND_PID" ]; then
    kill $BACKEND_PID
    echo "已停止后端服务，PID：$BACKEND_PID"
else
    echo "后端服务未运行"
fi

# 清理状态
echo "已清理运行状态"
echo "=============== 停止完成！ ==============="