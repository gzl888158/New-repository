"""
OKX 量化交易系统 - Python GUI 客户端
内置 tkinter，无需额外安装
"""
import os
import sys
import subprocess
import time
import json
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


class TradingClient:
    def __init__(self, root):
        self.root = root
        self.root.title("OKX 量化交易系统")
        self.root.geometry("520x560")
        self.root.resizable(False, False)

        self.core_proc = None
        self.dashboard_proc = None
        self.running = False
        self.monitor_thread = None

        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        # 标题
        tk.Label(self.root, text="OKX 量化交易系统", font=("Microsoft YaHei", 16, "bold")).pack(pady=10)

        # 状态面板
        frame = tk.Frame(self.root)
        frame.pack(pady=5, padx=20, fill="x")

        tk.Label(frame, text="运行状态:", font=("Microsoft YaHei", 10)).grid(row=0, column=0, sticky="w")
        self.lbl_status = tk.Label(frame, text="已停止", fg="red", font=("Microsoft YaHei", 10, "bold"))
        self.lbl_status.grid(row=0, column=1, sticky="w", padx=5)

        tk.Label(frame, text="当前权益:", font=("Microsoft YaHei", 10)).grid(row=1, column=0, sticky="w", pady=3)
        self.lbl_equity = tk.Label(frame, text="--", font=("Microsoft YaHei", 10))
        self.lbl_equity.grid(row=1, column=1, sticky="w", padx=5)

        tk.Label(frame, text="今日盈亏:", font=("Microsoft YaHei", 10)).grid(row=2, column=0, sticky="w", pady=3)
        self.lbl_pnl = tk.Label(frame, text="--", font=("Microsoft YaHei", 10))
        self.lbl_pnl.grid(row=2, column=1, sticky="w", padx=5)

        tk.Label(frame, text="持仓数量:", font=("Microsoft YaHei", 10)).grid(row=3, column=0, sticky="w", pady=3)
        self.lbl_positions = tk.Label(frame, text="--", font=("Microsoft YaHei", 10))
        self.lbl_positions.grid(row=3, column=1, sticky="w", padx=5)

        tk.Label(frame, text="风控状态:", font=("Microsoft YaHei", 10)).grid(row=4, column=0, sticky="w", pady=3)
        self.lbl_risk = tk.Label(frame, text="--", font=("Microsoft YaHei", 10))
        self.lbl_risk.grid(row=4, column=1, sticky="w", padx=5)

        # 按钮区
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        self.btn_start = tk.Button(btn_frame, text="启动系统", width=12, height=2, bg="#4CAF50", fg="white",
                                   font=("Microsoft YaHei", 10), command=self._start)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = tk.Button(btn_frame, text="停止系统", width=12, height=2, bg="#f44336", fg="white",
                                  font=("Microsoft YaHei", 10), command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        tk.Button(btn_frame, text="打开浏览器", width=12, height=2, bg="#2196F3", fg="white",
                  font=("Microsoft YaHei", 10), command=self._open_browser).pack(side="left", padx=5)

        # 资金分配面板
        alloc_frame = tk.LabelFrame(self.root, text="资金分配", font=("Microsoft YaHei", 10))
        alloc_frame.pack(pady=10, padx=20, fill="x")

        self.alloc_tree = ttk.Treeview(alloc_frame, columns=("strategy", "current", "optimal", "action"), show="headings", height=4)
        self.alloc_tree.heading("strategy", text="策略")
        self.alloc_tree.heading("current", text="当前分配")
        self.alloc_tree.heading("optimal", text="建议分配")
        self.alloc_tree.heading("action", text="操作建议")
        self.alloc_tree.column("strategy", width=100)
        self.alloc_tree.column("current", width=80)
        self.alloc_tree.column("optimal", width=80)
        self.alloc_tree.column("action", width=80)
        self.alloc_tree.pack(fill="x")

        alloc_btn_frame = tk.Frame(alloc_frame)
        alloc_btn_frame.pack(pady=5)
        tk.Button(alloc_btn_frame, text="刷新分配", width=10, command=self._refresh_allocation).pack(side="left", padx=5)
        tk.Button(alloc_btn_frame, text="手动重平衡", width=10, command=self._manual_rebalance).pack(side="left", padx=5)

        # 日志区
        tk.Label(self.root, text="运行日志:", font=("Microsoft YaHei", 9)).pack(anchor="w", padx=20, pady=(10, 0))
        self.txt_log = tk.Text(self.root, height=6, width=60, state="disabled", font=("Consolas", 9))
        self.txt_log.pack(padx=20, pady=5)

    def _log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", f"{time.strftime('%H:%M:%S')} {msg}\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    def _start(self):
        if self.running:
            return
        self._log("正在启动...")

        # 清理旧进程
        self._kill_old()
        lock = os.path.join(BASE_DIR, "data", "app.lock")
        if os.path.exists(lock):
            os.remove(lock)

        # 启动核心系统
        self.core_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        self._log(f"核心系统 PID: {self.core_proc.pid}")

        # 等待核心就绪
        hb = os.path.join(BASE_DIR, "data", "heartbeat.json")
        for i in range(60):
            if os.path.exists(hb):
                break
            time.sleep(1)
        else:
            self._log("核心系统启动超时")
            self._stop()
            return
        time.sleep(3)

        # 启动 Dashboard
        self.dashboard_proc = subprocess.Popen(
            [sys.executable, "dashboard_api.py"],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self._log(f"Dashboard PID: {self.dashboard_proc.pid}")

        # 等待端口
        import socket
        for i in range(30):
            try:
                with socket.create_connection(("127.0.0.1", 8080), timeout=1):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            self._log("Dashboard 启动超时")
            self._stop()
            return

        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="运行中", fg="green")
        self._log("系统已就绪")

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

        # 自动打开浏览器
        webbrowser.open("http://127.0.0.1:8080")

    def _stop(self):
        self._log("正在停止...")
        self.running = False

        if self.core_proc and self.core_proc.poll() is None:
            self.core_proc.terminate()
            try:
                self.core_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.core_proc.kill()

        if self.dashboard_proc and self.dashboard_proc.poll() is None:
            self.dashboard_proc.terminate()
            try:
                self.dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_proc.kill()

        lock = os.path.join(BASE_DIR, "data", "app.lock")
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except PermissionError:
                pass

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="已停止", fg="red")
        self.lbl_equity.config(text="--")
        self.lbl_pnl.config(text="--")
        self.lbl_positions.config(text="--")
        self.lbl_risk.config(text="--")
        self._log("已停止")

    def _kill_old(self):
        my_pid = os.getpid()
        if os.name == "nt":
            os.system(f'taskkill /f /fi "PID ne {my_pid}" /im python.exe 2>nul')
            os.system(f'taskkill /f /fi "PID ne {my_pid}" /im pythonw.exe 2>nul')
        time.sleep(1)

    def _monitor(self):
        while self.running:
            try:
                # 检查核心进程
                if self.core_proc and self.core_proc.poll() is not None:
                    self.root.after(0, lambda: self._log("核心进程已退出"))
                    self.root.after(0, self._stop)
                    break

                # 读取风控状态
                risk_path = os.path.join(BASE_DIR, "data", "risk_status.json")
                if os.path.exists(risk_path):
                    with open(risk_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    equity = data.get("current_equity", 0)
                    pnl = data.get("daily_pnl", 0)
                    paused = data.get("is_paused", False)

                    # 通过默认参数绑定值，避免闭包问题
                    self.root.after(0, lambda e=equity: self.lbl_equity.config(text=f"{e:.2f} USDT"))
                    self.root.after(0, lambda p=pnl: self.lbl_pnl.config(text=f"{p:+.2f} USDT"))
                    self.root.after(0, lambda pa=paused: self.lbl_risk.config(
                        text="已暂停" if pa else "正常", fg="red" if pa else "green"))

                # 读取持仓（带超时，避免数据库锁定）
                db_path = os.path.join(BASE_DIR, "data", "trading.db")
                if os.path.exists(db_path):
                    import sqlite3
                    conn = sqlite3.connect(db_path, timeout=5)
                    cursor = conn.execute("SELECT COUNT(*) FROM trade_records WHERE status='open'")
                    count = cursor.fetchone()[0]
                    conn.close()
                    self.root.after(0, lambda c=count: self.lbl_positions.config(text=f"{c}"))

            except Exception as e:
                self.root.after(0, lambda msg=str(e): self._log(f"同步错误: {msg}"))

            time.sleep(5)

    def _refresh_status(self):
        # 检查是否有已运行的进程
        import socket
        try:
            with socket.create_connection(("127.0.0.1", 8080), timeout=1):
                self.lbl_status.config(text="运行中(外部启动)", fg="orange")
        except Exception:
            pass

    def _open_browser(self):
        webbrowser.open("http://127.0.0.1:8080")

    def _refresh_allocation(self):
        try:
            alloc_path = os.path.join(BASE_DIR, "data", "allocation_state.json")
            if os.path.exists(alloc_path):
                with open(alloc_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                weights = state.get("strategy_weights", {})

                for item in self.alloc_tree.get_children():
                    self.alloc_tree.delete(item)

                for strategy, weight in weights.items():
                    self.alloc_tree.insert("", "end", values=(strategy, f"{weight:.2%}", f"{weight:.2%}", "hold"))
                self._log("资金分配已刷新")
            else:
                self._log("分配状态文件未找到")
        except Exception as e:
            self._log(f"刷新分配失败: {e}")

    def _manual_rebalance(self):
        try:
            alloc_path = os.path.join(BASE_DIR, "data", "allocation_state.json")
            if os.path.exists(alloc_path):
                with open(alloc_path, "r", encoding="utf-8") as f:
                    state = json.load(f)

                weights = state.get("strategy_weights", {})
                new_weights = self._calculate_equal_allocation(weights)

                for item in self.alloc_tree.get_children():
                    self.alloc_tree.delete(item)

                for strategy, weight in new_weights.items():
                    diff = weight - weights.get(strategy, 0)
                    action = "increase" if diff > 0.001 else "decrease" if diff < -0.001 else "hold"
                    self.alloc_tree.insert("", "end", values=(strategy, f"{weights.get(strategy, 0):.2%}", f"{weight:.2%}", action))

                self._log("手动重平衡建议已生成")
            else:
                self._log("分配状态文件未找到")
        except Exception as e:
            self._log(f"手动重平衡失败: {e}")

    def _calculate_equal_allocation(self, current_weights):
        strategies = list(current_weights.keys())
        equal_weight = 1.0 / len(strategies) if strategies else 0
        return {s: equal_weight for s in strategies}

    def on_close(self):
        if self.running:
            if messagebox.askyesno("确认", "系统正在运行，是否停止并退出？"):
                self._stop()
            else:
                return
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TradingClient(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
