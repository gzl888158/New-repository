"""
OKX 量化交易系统 - 停止脚本
将所有停止逻辑统一在 Python 中实现，批处理仅做最简转发。
"""
import os
import sys
import time
import socket
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── 工具函数 ──

def run_ps(script, timeout=15):
    """运行 PowerShell 脚本"""
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
    """获取本项目所有 Python 进程 PID（含 venv launcher 与 base python 子进程）。

    按父子关系识别，避免漏杀 watchdog/dashboard 的 base python（脚本参数为相对路径，
    不含项目路径）。
    """
    script = (
        """$all = @(Get-WmiObject Win32_Process -Filter "Name='python.exe'" """
        """-ErrorAction SilentlyContinue); """
        """$roots = @($all | Where-Object { $_.CommandLine -like '*okx_quant_trading*' """
        """-and $_.CommandLine -notlike '*Get-WmiObject*' """
        """-and $_.CommandLine -notlike '*stop.py*' """
        """-and $_.CommandLine -notlike '*health_check*' }); """
        """$rootIds = @($roots | ForEach-Object { [int]$_.ProcessId }); """
        """$proj = @($all | Where-Object { """
        """($_.CommandLine -like '*okx_quant_trading*' """
        """-and $_.CommandLine -notlike '*Get-WmiObject*' """
        """-and $_.CommandLine -notlike '*stop.py*' """
        """-and $_.CommandLine -notlike '*health_check*') """
        """-or ($rootIds -contains [int]$_.ParentProcessId) }); """
        """$proj | ForEach-Object { Write-Host $_.ProcessId }"""
    )
    _, stdout, _ = run_ps(script)
    if not stdout:
        return []
    return [int(p) for p in stdout.split()]

def check_port(port=8080):
    """检查端口是否被监听"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False
    finally:
        sock.close()

# ── 主流程 ──

def main():
    print("=" * 60)
    print("  OKX 量化交易系统 - 停止脚本")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # ── [1/4] 优雅关闭 ──
    print("[1/4] 优雅关闭（通知进程自行退出）...")
    pids = get_project_pids()
    if not pids:
        print("      无运行中进程，跳过")
    else:
        print(f"      通知 {len(pids)} 个进程退出...")
        # 先用 Stop-Process（无 -Force），相当于发送关闭信号
        kill_script = ";".join(
            f"Stop-Process -Id {pid} -ErrorAction SilentlyContinue"
            for pid in pids
        )
        run_ps(kill_script, timeout=10)

        # 轮询等待最多 10 秒
        print("      等待进程退出（最多 10 秒）...")
        all_gone = False
        for _ in range(10):
            time.sleep(1)
            remaining = get_project_pids()
            if not remaining:
                all_gone = True
                break
        if all_gone:
            print("      全部已退出")
        else:
            remaining = get_project_pids()
            print(f"      仍有 {len(remaining)} 个进程未退出")

    print()

    # ── [2/4] 强制清理 ──
    remaining = get_project_pids()
    if remaining:
        print("[2/4] 强制清理残留进程...")
        print(f"      强制终止 {len(remaining)} 个进程...")
        kill_script = ";".join(
            f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"
            for pid in remaining
        )
        run_ps(kill_script, timeout=10)
        time.sleep(3)

        # 复查
        final = get_project_pids()
        if final:
            print(f"      [WARN] {len(final)} 个顽固进程无法终止 (PID: {', '.join(map(str, final))})")
        else:
            print("      0 残留进程")
    else:
        print("[2/4] 跳过（无残留进程）")
    print()

    # ── [3/4] 清理残留文件 ──
    print("[3/4] 清理残留文件...")
    files_to_clean = [
        "data/app.lock",
        "data/heartbeat.json",
        "data/startup_status.json",
    ]
    for f in files_to_clean:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"      已清理: {f}")
            except Exception:
                # 如果普通删除失败，尝试 PowerShell 强制删除
                ps_path = f.replace("\\", "\\\\")
                run_ps(f"Remove-Item '{ps_path}' -Force -ErrorAction SilentlyContinue", timeout=5)

    # 清理 _workspace 目录
    ws_dir = os.path.join(BASE_DIR, "data", "_workspace")
    if os.path.exists(ws_dir):
        import shutil
        try:
            shutil.rmtree(ws_dir, ignore_errors=True)
            print("      已清理: data/_workspace")
        except Exception:
            pass
    print()

    # ── [4/4] 验证端口释放 ──
    print("[4/4] 验证端口释放...")
    if check_port(8080):
        # 端口仍被占用，尝试查出占用者
        port_script = (
            "$conn = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue"
            " | Where-Object { $_.State -eq 'Listen' };"
            "if ($conn) {"
            "  $p = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue;"
            "  Write-Host ('[WARN] 端口 8080 仍被占用: ' + $p.ProcessName + ' (PID=' + $conn.OwningProcess + ')')"
            "} else { Write-Host '端口 8080: 已释放' }"
        )
        run_ps(port_script)
    else:
        print("      端口 8080: 已释放")
    print()

    # ── 总结 ──
    print("=" * 60)
    print("  系统已停止")
    print("=" * 60)
    print()

    final_pids = get_project_pids()
    if not final_pids:
        print("  [OK] 无本项目残留进程")
        print("  [OK] 可以安全启动")
    else:
        print(f"  [!!] 仍有 {len(final_pids)} 个残留进程，建议重启电脑后再启动")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            input("按 Enter 退出...")
        except (EOFError, OSError):
            pass
