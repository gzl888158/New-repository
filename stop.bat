@echo off
echo =============== 欧易网格交易机器人 一键停止 ===============
echo 停止后端服务...
for /f "tokens=2 delims= " %%a in ('tasklist ^| findstr /i "python.exe" ^| findstr /i "src.main"') do (
    taskkill /f /pid %%a
    echo 已停止后端服务，PID：%%a
)
echo =============== 停止完成！ ===============
pause