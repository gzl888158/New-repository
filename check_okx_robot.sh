#!/bin/bash
echo "=============================================="
echo "欧易网格机器人 本地环境自检工具（Linux/Mac版）"
echo "检测时间：$(date)"
echo "=============================================="
echo

# 定义核心文件/目录列表
required_dirs=("config" "src" "logs")
required_config="config/config.yaml"
required_src=("src/__init__.py" "src/main.py" "src/okx_api.py" "src/utils.py")
required_root=("start.bat" "stop.bat" "start.sh" "stop.sh" ".env" "index.html" "requirements.txt")
port=8000
error=0

# 1. 检查核心目录是否存在
echo "【1】核心目录检查"
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "[✅] 目录 $dir 存在"
    else
        echo "[❌] 目录 $dir 缺失！请创建"
        error=1
    fi
done
echo

# 2. 检查核心文件是否存在
echo "【2】核心文件检查"
# config目录
if [ -f "$required_config" ]; then
    echo "[✅] $required_config 存在"
else
    echo "[❌] $required_config 缺失！请创建并粘贴配置"
    error=1
fi
# src目录
for file in "${required_src[@]}"; do
    if [ -f "$file" ]; then
        echo "[✅] $file 存在"
    else
        echo "[❌] $file 缺失！请创建并粘贴代码"
        error=1
    fi
done
# 根目录
for file in "${required_root[@]}"; do
    if [ -f "$file" ]; then
        echo "[✅] $file 存在"
    else
        echo "[⚠️] $file 缺失（Linux/Mac可忽略bat脚本）"
    fi
done
echo

# 3. 检查端口是否被占用
echo "【3】后端端口（$port）占用检查"
if lsof -i :$port >/dev/null; then
    echo "[⚠️] 端口 $port 被占用！进程信息："
    lsof -i :$port | grep LISTEN
    echo "请执行 kill -9 <PID> 结束进程后重启"
else
    echo "[✅] 端口 $port 未被占用"
fi
echo

# 4. 检查后端服务是否运行
echo "【4】后端服务运行状态检查"
pid=$(ps aux | grep "src.main" | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo "[✅] 后端服务正在运行，PID：$pid"
else
    echo "[❌] 后端服务未运行！请执行 ./start.sh 启动"
    error=1
fi
echo

# 5. 自检结果汇总
echo "=============================================="
if [ $error -eq 1 ]; then
    echo "[❌] 自检未通过！请修复上述问题后重试"
else
    echo "[✅] 自检通过！请本地打开 index.html 开始测试"
fi