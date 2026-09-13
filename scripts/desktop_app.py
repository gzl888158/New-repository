"""
OKX 量化交易系统 - 独立桌面启动器
一键启动：核心交易系统 + Dashboard API + 桌面窗口
架构：服务层设计（SignalProcessor + TradingSchedulerService）
"""
import os
import sys
import threading
import time
import subprocess
import signal
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

_core_process = None
_flask_thread = None


def _get_resource_path(relative_path: str) -> str:
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = BASE_DIR
    return os.path.join(base, relative_path)


def _check_port(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0


def start_core_system():
    global _core_process
    print("[Launcher] 启动核心交易系统 (Service Layer)...")

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    _core_process = subprocess.Popen(
        [sys.executable, 'main.py'],
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )

    def log_forward():
        for line in _core_process.stdout:
            print(f"[Core] {line.rstrip()}")

    threading.Thread(target=log_forward, daemon=True).start()
    print(f"[Launcher] 核心系统 PID: {_core_process.pid}")


def start_flask():
    print("[Launcher] 启动 Dashboard API...")
    import dashboard_api
    dashboard_api.load_config()
    dashboard_api.app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)


def wait_for_server(url: str, timeout: int = 30):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def wait_for_core_ready(timeout: int = 60):
    heartbeat_path = os.path.join(BASE_DIR, 'data', 'heartbeat.json')
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(heartbeat_path):
            try:
                import json
                with open(heartbeat_path, 'r') as f:
                    hb = json.load(f)
                if hb.get('status') in ('ok', 'running'):
                    return True
            except Exception:
                pass
        time.sleep(1)
    return False


def cleanup():
    print("\n[Launcher] 正在关闭服务...")

    if _core_process and _core_process.poll() is None:
        print(f"[Launcher] 终止核心系统 PID {_core_process.pid}")
        if os.name == 'nt':
            _core_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            _core_process.send_signal(signal.SIGTERM)
        try:
            _core_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _core_process.kill()

    time.sleep(0.5)
    lock_file = os.path.join(BASE_DIR, 'data', 'app.lock')
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except PermissionError:
            pass

    print("[Launcher] 已清理完成")


def main():
    global _core_process, _flask_thread

    print("=" * 60)
    print("  OKX 量化交易系统 - 桌面启动器")
    print("  架构: Service Layer (SignalProcessor)")
    print("=" * 60)

    atexit.register(cleanup)

    if not _check_port(8080):
        print("[Launcher] 警告: 端口 8080 已被占用，尝试关闭旧进程...")
        if os.name == 'nt':
            os.system("taskkill /f /im python.exe 2>nul")
            os.system("taskkill /f /im pythonw.exe 2>nul")
        time.sleep(2)

    lock_file = os.path.join(BASE_DIR, 'data', 'app.lock')
    if os.path.exists(lock_file):
        os.remove(lock_file)

    print("[Launcher] Step 1/4: 启动核心交易系统...")
    start_core_system()

    print("[Launcher] Step 2/4: 等待核心系统初始化...")
    if not wait_for_core_ready(timeout=60):
        print("[Launcher] 核心系统启动超时")
        sys.exit(1)
    print("[Launcher] 核心系统已就绪")

    print("[Launcher] Step 3/4: 启动 Dashboard API...")
    _flask_thread = threading.Thread(target=start_flask, daemon=True)
    _flask_thread.start()

    url = "http://127.0.0.1:8080"
    print(f"[Launcher] Step 4/4: 等待 Dashboard 就绪 {url}...")
    if not wait_for_server(url):
        print("[Launcher] Dashboard 启动超时")
        sys.exit(1)
    print("[Launcher] Dashboard 已就绪")

    print("[Launcher] 正在打开浏览器...")
    import webbrowser
    webbrowser.open(url)

    print("[Launcher] 系统运行中，关闭此窗口将停止所有服务")
    try:
        while True:
            time.sleep(1)
            if _core_process and _core_process.poll() is not None:
                print("[Launcher] 核心进程已退出，正在关闭...")
                break
    except KeyboardInterrupt:
        print("\n[Launcher] 收到中断信号")


if __name__ == '__main__':
    main()
