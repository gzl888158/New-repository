"""
OKX 量化交易系统 - 健康检查脚本
一键诊断：进程 → HTTP → 心跳 → 文件 → 错误 → 总结
"""
import os
import sys
import json
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DASHBOARD_URL = "http://localhost:8080"
HEALTH_URL = f"{DASHBOARD_URL}/api/health"


def run_ps(script, timeout=15):
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


def get_project_pids():
    script = (
        """$procs = Get-WmiObject Win32_Process -Filter "Name='python.exe'" """
        """| Where-Object { ($_.CommandLine -like '*okx_quant_trading*' """
        """-or $_.CommandLine -like '*main.py*' """
        """-or $_.CommandLine -like '*dashboard_api.py*') """
        """-and $_.CommandLine -notlike '*Get-WmiObject*' """
        """-and $_.CommandLine -notlike '*health_check*' """
        """}; $procs | ForEach-Object { Write-Host $_.ProcessId }"""
    )
    _, stdout, _ = run_ps(script)
    if not stdout:
        return []
    return [int(p) for p in stdout.split()]


def get_http_json(url, timeout=5):
    import urllib.request
    try:
        req = urllib.request.Request(url, method="GET")
        req.timeout = timeout
        with urllib.request.urlopen(req) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception:
        pass
    return None


def check_processes():
    """[1/5] 进程状态"""
    print("[1/5] 进程状态")
    script = (
        """$procs = Get-WmiObject Win32_Process -Filter "Name='python.exe'" """
        """| Where-Object { ($_.CommandLine -like '*okx_quant_trading*' """
        """-or $_.CommandLine -like '*main.py*' """
        """-or $_.CommandLine -like '*dashboard_api.py*') """
        """-and $_.CommandLine -notlike '*Get-WmiObject*' """
        """-and $_.CommandLine -notlike '*health_check*' """
        """}; """
        """if (@($procs).Count -gt 0) { """
        """  Write-Host ('  项目进程数: ' + @($procs).Count); """
        """  $procs | ForEach-Object { """
        """    $cmd = $_.CommandLine; """
        """    if ($cmd -like '*watchdog*') { Write-Host ('    watchdog      PID= ' + $_.ProcessId) } """
        """    elseif ($cmd -like '*main.py*') { Write-Host ('    main.py        PID= ' + $_.ProcessId) } """
        """    elseif ($cmd -like '*dashboard*') { Write-Host ('    dashboard      PID= ' + $_.ProcessId) } """
        """    else { Write-Host ('    other          PID= ' + $_.ProcessId) } """
        """  } """
        """} else { Write-Host '  无本项目进程' }"""
    )
    _, stdout, _ = run_ps(script)
    if stdout:
        print(stdout)
    pids = get_project_pids()
    print()
    return len(pids)


def check_dashboard_http():
    """[2/5] Dashboard HTTP，返回 True 表示可访问"""
    print("[2/5] Dashboard HTTP")
    data = get_http_json(HEALTH_URL, timeout=5)
    if data:
        print("  HTTP 200 OK")
        print(f"  Dashboard        : {data.get('dashboard', '?')}")
        print(f"  交易引擎         : {data.get('trading_engine', '?')}")
        if data.get("heartbeat_age_seconds") is not None:
            print(f"  心跳延迟         : {data['heartbeat_age_seconds']}s")
        if data.get("watchdog_phase"):
            print(f"  Watchdog 状态    : {data['watchdog_phase']}")
        if data.get("restart_count"):
            print(f"  今日重启次数     : {data['restart_count']}")
        print()
        return True
    print("  [FAIL] Dashboard 无响应")
    print()
    return False


def check_heartbeat(running):
    """[3/5] 交易引擎心跳"""
    print("[3/5] 交易引擎心跳")
    hb_path = os.path.join(BASE_DIR, "data", "heartbeat.json")
    if os.path.exists(hb_path):
        try:
            with open(hb_path, "r", encoding="utf-8") as f:
                hb = json.load(f)
            last_update = hb.get("last_update", "")
            if last_update:
                age = (datetime.now() - datetime.fromisoformat(str(last_update))).total_seconds()
                if age < 60:
                    print(f"  心跳: 正常 (age={int(age)}s) - 调度器运行中")
                elif age < 120:
                    print(f"  心跳: 延迟 (age={int(age)}s) - 可能网络延迟")
                else:
                    print(f"  心跳: 过期 (age={int(age)}s) - 引擎可能卡死!")
            else:
                print("  心跳文件缺少 last_update 字段")
        except Exception:
            print("  心跳文件存在但无法解析")
    else:
        if running:
            print("  心跳文件不存在 - 引擎刚启动或写入异常")
        else:
            print("  心跳文件不存在 - 引擎未运行")
    print()


def check_files():
    """[4/5] 文件状态"""
    print("[4/5] 文件状态")

    def status(name):
        path = os.path.join(BASE_DIR, name)
        return "存在" if os.path.exists(path) else "不存在"

    print(f"  app.lock: {status('data/app.lock')}")
    print(f"  heartbeat.json: {status('data/heartbeat.json')}")

    status_path = os.path.join(BASE_DIR, "data", "startup_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            print(f"  startup_status.json: 存在  (phase: {s.get('phase', '?')})")
        except Exception:
            print("  startup_status.json: 存在 (无法解析)")
    else:
        print("  startup_status.json: 不存在")
    print()


def check_errors():
    """[5/5] 最近错误"""
    print("[5/5] 最近错误（最近 5 条）")
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(log_dir):
        print("  无错误日志")
        print()
        return

    # 找最新的 error_ 日志
    error_logs = []
    for f in os.listdir(log_dir):
        if f.startswith("error_"):
            error_logs.append(os.path.join(log_dir, f))
    if not error_logs:
        print("  无错误日志")
        print()
        return

    latest = max(error_logs, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # 取最后 5 条非空行
        lines = [l.strip() for l in lines if l.strip()]
        for line in lines[-5:]:
            print(f"  {line}")
    except Exception as e:
        print(f"  无法读取: {e}")
    print()


def collect_health_status():
    """汇总健康状态为结构化 dict，供 --json 模式与自动化复用。"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "dashboard_url": DASHBOARD_URL,
    }

    # 进程
    pids = get_project_pids()
    status["project_process_count"] = len(pids)
    status["project_pids"] = pids

    # Dashboard HTTP
    data = get_http_json(HEALTH_URL, timeout=5)
    status["dashboard_http"] = data is not None
    status["dashboard_health"] = data

    # 心跳
    hb_path = os.path.join(BASE_DIR, "data", "heartbeat.json")
    if os.path.exists(hb_path):
        try:
            with open(hb_path, "r", encoding="utf-8") as f:
                hb = json.load(f)
            last_update = hb.get("last_update", "")
            if last_update:
                age = (datetime.now() - datetime.fromisoformat(str(last_update))).total_seconds()
                status["heartbeat_age_seconds"] = int(age)
                status["heartbeat_status"] = "ok" if age < 60 else ("delayed" if age < 120 else "stale")
            status["heartbeat_pid"] = hb.get("pid")
            status["heartbeat_engine_status"] = hb.get("status")
        except Exception:
            status["heartbeat_error"] = "unparseable"
    else:
        status["heartbeat_status"] = "missing"

    # 关键文件
    status["files"] = {
        "app.lock": os.path.exists(os.path.join(BASE_DIR, "data", "app.lock")),
        "heartbeat.json": os.path.exists(hb_path),
    }
    status_path = os.path.join(BASE_DIR, "data", "startup_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status["startup_status"] = json.load(f)
        except Exception:
            status["startup_status"] = "unparseable"

    # 最近错误
    log_dir = os.path.join(BASE_DIR, "logs")
    recent_errors = []
    if os.path.exists(log_dir):
        error_logs = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("error_")]
        if error_logs:
            latest = max(error_logs, key=os.path.getmtime)
            try:
                with open(latest, "r", encoding="utf-8", errors="replace") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                recent_errors = lines[-5:]
            except Exception:
                pass
    status["recent_errors"] = recent_errors
    return status


def main():
    if "--json" in sys.argv:
        print(json.dumps(collect_health_status(), ensure_ascii=False, indent=2))
        return

    print("=" * 60)
    print("  OKX 量化交易系统 - 健康检查")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # [1] 进程检查
    proc_count = check_processes()

    # [2] Dashboard HTTP
    dash_ok = check_dashboard_http()

    # [3] 心跳检查
    check_heartbeat(proc_count > 0)

    # [4] 文件状态
    check_files()

    # [5] 最近错误
    check_errors()

    # ── 总结 ──
    print("=" * 60)
    print("  健康检查结果")
    print("=" * 60)

    if proc_count == 0:
        print("  [!!] 系统未运行 - 双击 启动交易系统.bat 启动")
    else:
        print("  [OK] 交易引擎: 进程运行中")
        if dash_ok:
            print("  [OK] Dashboard: 可访问")
        else:
            print("  [!!] Dashboard: 无响应")
        print(f"  [OK] 总进程数: {proc_count}")

    print(f"  Dashboard URL: {DASHBOARD_URL}")
    print(f"  API Health   : {HEALTH_URL}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            input("按 Enter 退出...")
        except (EOFError, OSError):
            pass
