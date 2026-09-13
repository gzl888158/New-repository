"""
OKX 量化交易系统 - 一键启动程序
双击运行即可启动完整交易环境
"""
import os
import sys
import subprocess
import time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def run_cmd(cmd, cwd=None, daemon=False):
    """运行命令"""
    kwargs = {
        "cwd": cwd or BASE_DIR,
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    }
    if daemon:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.Popen(cmd, **kwargs)


def kill_python():
    """关闭旧 Python 进程（排除当前进程）"""
    if os.name == "nt":
        my_pid = os.getpid()
        os.system(f'taskkill /f /fi "PID ne {my_pid}" /im python.exe 2>nul')
        os.system(f'taskkill /f /fi "PID ne {my_pid}" /im pythonw.exe 2>nul')
    time.sleep(1)


def wait_port(port, timeout=30):
    """等待端口就绪"""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print("=" * 50)
    print("  OKX 量化交易系统 - 一键启动")
    print("=" * 50)

    # 关闭旧进程
    print("[1/4] 清理旧进程...")
    kill_python()
    lock = os.path.join(BASE_DIR, "data", "app.lock")
    if os.path.exists(lock):
        os.remove(lock)

    # 启动核心系统
    print("[2/4] 启动核心交易系统...")
    core = run_cmd([sys.executable, "main.py"])

    # 等待核心就绪
    print("[3/4] 等待系统初始化...")
    hb = os.path.join(BASE_DIR, "data", "heartbeat.json")
    for _ in range(60):
        if os.path.exists(hb):
            break
        time.sleep(1)
    else:
        print("初始化超时，请检查日志")
        core.kill()
        return
    time.sleep(3)

    # 启动 Dashboard
    print("[4/4] 启动 Dashboard...")
    dashboard = run_cmd([sys.executable, "dashboard_api.py"], daemon=True)
    if not wait_port(8080):
        print("Dashboard 启动超时")
        core.kill()
        return

    # 打开浏览器
    url = "http://127.0.0.1:8080"
    print(f"\n系统已就绪: {url}")
    webbrowser.open(url)

    print("按 Ctrl+C 停止所有服务\n")
    try:
        while True:
            time.sleep(1)
            if core.poll() is not None:
                print("核心进程已退出")
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在停止服务...")
        core.terminate()
        dashboard.terminate()
        if os.path.exists(lock):
            os.remove(lock)
        print("已退出")


if __name__ == "__main__":
    main()
