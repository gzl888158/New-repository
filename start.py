"""
OKX 量化交易系统 - 核心启动编排器 v7
取代 start.bat 中的所有复杂批处理逻辑，统一用 Python 实现。
本机适配：Python 3.12.4 / Windows 10 / i7-1185G7 / 32GB RAM
"""
import os
import sys
import time
import json
import socket
import subprocess
import webbrowser
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── 加载 .env（最先执行） ─────────────────────────────────

def load_dotenv():
    """加载 .env 文件中的环境变量，优先级：已存在环境变量 > .env > 无"""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return False
    try:
        from dotenv import load_dotenv as _load
        _load(env_path, override=False)  # override=False: 不覆盖已有环境变量
        return True
    except ImportError:
        # 手动解析 .env（不依赖 python-dotenv）
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            return True
        except Exception:
            return False

load_dotenv()

# ── 工具函数 ──────────────────────────────────────────────

def log(msg=""):
    """即时输出，强制刷新"""
    print(msg, flush=True)

def run_powershell(script, timeout=30):
    """运行 PowerShell 脚本并返回 (exit_code, stdout, stderr)"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, timeout=timeout, cwd=BASE_DIR
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)

def get_project_processes():
    """返回本项目所有 Python 进程的 PID 列表（含 venv launcher 与 base python 子进程）。

    按父子关系识别：先按完整路径匹配项目 launcher 进程，再纳入这些进程的
    直接子进程——watchdog/dashboard 的 base python 进程脚本参数为相对路径
    （watchdog.py / dashboard_api.py），不含项目路径，仅按 CommandLine 匹配会漏杀。
    """
    script = (
        '$all = @(Get-WmiObject Win32_Process -Filter "Name=\'python.exe\'" '
        '-ErrorAction SilentlyContinue); '
        '$roots = @($all | Where-Object { $_.CommandLine -like \'*okx_quant_trading*\' '
        '-and $_.CommandLine -notlike \'*Get-WmiObject*\' '
        '-and $_.CommandLine -notlike \'*start.py*\' '
        '-and $_.CommandLine -notlike \'*stop.py*\' '
        '-and $_.CommandLine -notlike \'*health_check*\' }); '
        '$rootIds = @($roots | ForEach-Object { [int]$_.ProcessId }); '
        '$proj = @($all | Where-Object { '
        '($_.CommandLine -like \'*okx_quant_trading*\' '
        '-and $_.CommandLine -notlike \'*Get-WmiObject*\' '
        '-and $_.CommandLine -notlike \'*start.py*\' '
        '-and $_.CommandLine -notlike \'*stop.py*\' '
        '-and $_.CommandLine -notlike \'*health_check*\') '
        '-or ($rootIds -contains [int]$_.ParentProcessId) }); '
        '$proj | ForEach-Object { Write-Host $_.ProcessId }'
    )
    _, stdout, _ = run_powershell(script)
    if not stdout:
        return []
    return [int(p) for p in stdout.split()]

def is_main_py_alive():
    """批量检测是否有 main.py 进程（一次 PowerShell 调用）"""
    script = (
        '$procs = Get-WmiObject Win32_Process -Filter "Name=\'python.exe\'" '
        '| Where-Object { $_.CommandLine -like \'*okx_quant_trading*main.py*\' '
        '-and $_.CommandLine -notlike \'*Get-WmiObject*\' }; '
        'if ($procs) { Write-Host $procs[0].ProcessId; exit 0 } else { exit 1 }'
    )
    code, stdout, _ = run_powershell(script)
    if code == 0 and stdout.strip():
        return int(stdout.strip())
    return None

def kill_old_processes():
    """清理由本项目残留的 Python 进程（watchdog/main/dashboard）"""
    my_pid = os.getpid()
    pids = get_project_processes()
    pids = [p for p in pids if p != my_pid]
    if not pids:
        log("  无残留进程")
        return True
    log(f"  发现 {len(pids)} 个残留进程: {pids} (当前 PID={my_pid})")
    for pid in pids:
        if pid == my_pid:
            continue
        try:
            run_powershell(f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue", timeout=5)
        except Exception:
            pass
    time.sleep(2)
    remaining = [p for p in get_project_processes() if p != my_pid]
    if remaining:
        log(f"  [WARN] {len(remaining)} 个顽固残留: {remaining}")
        return False
    log(f"  已清理 {len(pids)} 个进程，0 残留")
    return True

def wait_port_free(port=8080, timeout=15):
    """等待端口释放，返回 True 表示空闲"""
    if not check_port(port):
        return True
    log(f"  等待端口 {port} 释放（最多 {timeout}s）...")
    deadline = time.time() + timeout
    while check_port(port) and time.time() < deadline:
        time.sleep(0.5)
    if check_port(port):
        log(f"  [WARN] 端口 {port} 未释放（TIME_WAIT），使用 SO_REUSEADDR 强制绑定")
        return False
    log(f"  端口 {port} 已释放")
    return True

def hard_stop():
    """彻底停止：多轮杀进程（防 watchdog 重新拉起）+ 清理锁 + 等待端口"""
    my_pid = os.getpid()
    log("  彻底停止所有进程...")
    # 多轮循环：防止 watchdog 在杀 main.py 后立即重新拉起
    for attempt in range(4):
        pids = [p for p in get_project_processes() if p != my_pid]
        if not pids:
            break
        for pid in pids:
            run_powershell(f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue", timeout=5)
        if attempt < 3:
            time.sleep(1.5)
    # 验证
    remaining = [p for p in get_project_processes() if p != my_pid]
    if remaining:
        log(f"  [WARN] {len(remaining)} 个顽固残留: {remaining}")
    else:
        log("  进程已全部停止")
    clean_lock_files()
    wait_port_free(8080, timeout=15)

def check_port(port=8080):
    """检查端口是否被监听"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        return result == 0
    except Exception:
        return False
    finally:
        sock.close()

def wait_http(url, timeout=60, interval=2):
    """轮询等待 HTTP 200 响应"""
    import urllib.request
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            req.timeout = 2
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
        dots += 1
        if dots % 5 == 0:
            elapsed = dots * interval
            log(f"    等待中... ({elapsed}s / {timeout}s)")
    return False

def clean_lock_files():
    """清理锁文件、心跳文件、临时目录"""
    for f in ["data/app.lock", "data/heartbeat.json", "data/startup_status.json"]:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            try:
                os.remove(path)
                log(f"  已清理: {f}")
            except Exception:
                pass
    ws = os.path.join(BASE_DIR, "data", "_workspace")
    if os.path.exists(ws):
        try:
            import shutil
            shutil.rmtree(ws, ignore_errors=True)
        except Exception:
            pass

# ── 主流程 ──────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("  OKX 量化交易系统 - 启动编排器 v7")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # ── 前置检查：系统是否已在运行 ──
    skip_cleanup = False  # 重启路径跳过 [1/5][2/5]
    running_pids = get_project_processes()
    if running_pids:
        dash_accessible = check_port(8080)
        print("  系统已在运行中!")
        print(f"  进程 PID: {running_pids}")
        if dash_accessible:
            print("  Dashboard: http://localhost:8080 (可访问)")
        else:
            print("  Dashboard: http://localhost:8080 (等待中...)")
        print()
        print("  [1] 打开 Dashboard")
        print("  [2] 重启系统")
        print("  [3] 停止系统")
        print("  [0] 退出")
        print()
        try:
            choice = input("请选择 [1/2/3/0]: ").strip()
        except (EOFError, OSError):
            choice = "0"
        if choice == "1":
            try: webbrowser.open("http://localhost:8080")
            except Exception: pass
            sys.exit(0)
        elif choice == "2":
            print()
            hard_stop()
            skip_cleanup = True
            print()
            print("  正在重新启动...")
            print()
            # 继续正常启动流程，但跳过 [1/5][2/5]
        elif choice == "3":
            print()
            hard_stop()
            print()
            print("  系统已停止")
            try: input("按 Enter 退出...")
            except (EOFError, OSError): pass
            sys.exit(0)
        else:
            sys.exit(0)
    print()

    # ── [0/5] 环境预检 ──
    log("[0/5] 环境预检")
    checks = {
        ".venv/Scripts/python.exe": "虚拟环境",
        "config.yaml": "配置文件",
        "main.py": "交易主程序",
        "dashboard_api.py": "Dashboard",
    }
    for path, name in checks.items():
        if not os.path.exists(os.path.join(BASE_DIR, path)):
            log(f"  [FAIL] {name} 不存在: {path}")
            try: input("\n按 Enter 退出...")
            except (EOFError, OSError): pass
            sys.exit(1)
    log("  文件检查: OK")

    # 端口检测
    if check_port(8080):
        pids = get_project_processes()
        if pids:
            log(f"  端口 8080: 本项目进程占用 (PID: {pids})，清理后会释放")
        else:
            log("  [WARN] 端口 8080 已被外部进程占用")
    else:
        log("  端口 8080: 空闲")

    # 磁盘检查
    try:
        import shutil
        free = shutil.disk_usage(BASE_DIR).free / (1024**3)
        if free < 0.5:
            log(f"  [WARN] 磁盘剩余空间: {free:.1f}GB")
        else:
            log(f"  磁盘剩余空间: {free:.1f}GB")
    except Exception:
        pass

    # .env / 环境变量检查
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        log(f"  .env 文件: 已加载 ({os.path.getsize(env_path)} bytes)")
    missing_keys = []
    for key in ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]:
        if key not in os.environ:
            missing_keys.append(key)
    if missing_keys:
        log(f"  [WARN] 缺少环境变量: {', '.join(missing_keys)}")
        if os.path.exists(env_path):
            log("         已从 .env 加载但似乎未生效，请检查 .env 格式")
        else:
            log("         请创建 .env 文件并填入 OKX API 密钥")
    else:
        log("  环境变量: OK (API密钥已配置)")
    log()

    # ── [1/5] 清理旧进程（重启时跳过）──
    if skip_cleanup:
        log("[1/5] 清理旧进程   (已完成)")
        log("[2/5] 清理锁文件   (已完成)")
        log()
    else:
        log("[1/5] 清理旧进程")
        kill_old_processes()
        log()
        log("[2/5] 清理锁文件")
        clean_lock_files()
        log()

    # ── [3/5] 配置预检 ──
    log("[3/5] 配置预检")
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "from configs.settings import load_config; cfg=load_config(); "
             "print(f'{len(cfg.get(\"strategies\",{}))}|"
             "{cfg.get(\"system\",{}).get(\"version\",\"?\")}')"],
            capture_output=True, text=True, timeout=15, cwd=BASE_DIR
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split("|")
            n_strat = parts[0] if parts else "?"
            sys_ver = parts[1] if len(parts) > 1 else "?"
            log(f"  配置加载成功, {n_strat} 个策略")
            log(f"  系统版本: v{sys_ver}")
        else:
            log(f"  [FAIL] 配置验证失败:\n{r.stderr}")
            try: input("\n按 Enter 退出...")
            except (EOFError, OSError): pass
            sys.exit(1)
    except Exception as e:
        log(f"  [FAIL] 配置验证异常: {e}")
        try: input("\n按 Enter 退出...")
        except (EOFError, OSError): pass
        sys.exit(1)
    log()

    # ── [4/5] 启动 Watchdog ──
    log("[4/5] 启动核心交易引擎")
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    # 写启动状态
    try:
        with open(os.path.join(BASE_DIR, "data", "startup_status.json"), "w", encoding="utf-8") as f:
            json.dump({"phase": "watchdog_start", "time": datetime.now().isoformat()}, f)
    except Exception:
        pass

    # 启动 watchdog
    log("  启动 Watchdog...")
    wd = subprocess.Popen(
        [sys.executable, "watchdog.py"],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    log(f"  Watchdog PID: {wd.pid}")

    # 轮询等待 main.py 进程出现（批量检测，一次 PowerShell 调用）
    log("  等待交易引擎初始化...")
    main_pid = None
    for i in range(24):  # 最多 48 秒
        time.sleep(2)
        main_pid = is_main_py_alive()
        if main_pid:
            log(f"  交易引擎就绪 (PID: {main_pid}, {((i+1)*2)}s)")
            break
        sys.stdout.write("."); sys.stdout.flush()
    if not main_pid:
        log("\n  [WARN] 交易引擎未在 48 秒内就绪")
        log("         watchdog 可能仍在重试，查看 logs/watchdog_*.log")
    log()

    # ── [5/5] 启动 Dashboard ──
    log("[5/5] 启动 Dashboard")
    wait_port_free(8080, timeout=15)

    dash = subprocess.Popen(
        [sys.executable, "dashboard_api.py"],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    log(f"  Dashboard PID: {dash.pid}")

    log("  等待 Dashboard HTTP 就绪（最长 90 秒）...")
    dash_ok = wait_http("http://127.0.0.1:8080/api/health", timeout=90, interval=2)
    log()

    # ── 写最终状态 ──
    try:
        status = {
            "phase": "running",
            "start_time": datetime.now().isoformat(),
            "dashboard_url": "http://localhost:8080",
            "dashboard_port": 8080,
            "main_ready": main_pid is not None,
            "dashboard_ready": dash_ok,
            "startup_elapsed_seconds": int(time.time() - t_start),
        }
        with open(os.path.join(BASE_DIR, "data", "startup_status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # ── 自动打开浏览器 ──
    try:
        webbrowser.open("http://localhost:8080")
    except Exception:
        pass

    # ── 启动完成 ──
    elapsed = int(time.time() - t_start)
    print("=" * 60)
    print("  启动完成!  总耗时: {}秒".format(elapsed))
    print()
    print("  Dashboard  : http://localhost:8080")
    print("  交易引擎   : 由 Watchdog 守护（崩溃自动重启）")
    print("  关闭方式   : 双击 stop.bat")
    print("  诊断工具   : health_check.bat")
    print("=" * 60)
    print()

    if main_pid:
        print("  [OK] 交易引擎  运行中  (PID: {})".format(main_pid))
    else:
        print("  [!!] 交易引擎  未就绪")
    if dash_ok:
        print("  [OK] Dashboard  可访问")
    else:
        print("  [!!] Dashboard  无响应（稍后手动访问或运行 health_check.bat）")

    print()
    try:
        input("按 Enter 关闭此窗口（系统将在后台继续运行）...")
    except (EOFError, OSError):
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n用户中断")
    except EOFError:
        log("\n终端已关闭（系统已在后台继续运行）")
    except Exception as e:
        log(f"\n[FATAL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            input("\n按 Enter 退出...")
        except (EOFError, OSError):
            pass
