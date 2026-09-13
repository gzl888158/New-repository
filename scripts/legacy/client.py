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
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DASHBOARD_URL = "http://127.0.0.1:8080"
DASHBOARD_TIMEOUT = 3  # 秒，HTTP 请求超时


# ============================================================================
# HTTP 工具
# ============================================================================

def api_get(path, timeout=DASHBOARD_TIMEOUT):
    """GET 请求 dashboard API，返回 dict 或 None"""
    try:
        url = f"{DASHBOARD_URL}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def api_post(path, payload=None, timeout=DASHBOARD_TIMEOUT):
    """POST 请求 dashboard API，返回 (success: bool, data: dict)"""
    try:
        url = f"{DASHBOARD_URL}{path}"
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return (not result.get("error")), result
    except urllib.error.URLError as e:
        return False, {"error": f"无法连接 Dashboard: {e}"}
    except Exception as e:
        return False, {"error": str(e)}


# ============================================================================
# 主客户端
# ============================================================================

class TradingClient:
    def __init__(self, root):
        self.root = root
        self.root.title("OKX 量化交易系统")
        self.root.geometry("920x860")
        self.root.resizable(False, False)

        self.core_proc = None
        self.dashboard_proc = None
        self.running = False
        self.monitor_thread = None

        # 配色
        self.c_green = "#27ae60"
        self.c_red = "#e74c3c"
        self.c_orange = "#f39c12"
        self.c_blue = "#2980b9"
        self.c_purple = "#8e44ad"
        self.c_gray = "#7f8c8d"
        self.c_bg = "#f5f6fa"

        self.root.configure(bg=self.c_bg)
        self._build_ui()
        self._refresh_status()

    # ====================================================================
    # UI 构建
    # ====================================================================

    def _build_ui(self):
        # 顶部标题栏
        header = tk.Frame(self.root, bg=self.c_blue, height=46)
        header.pack(fill="x")
        tk.Label(
            header, text="OKX 量化交易系统", font=("Microsoft YaHei", 15, "bold"),
            fg="white", bg=self.c_blue
        ).pack(side="left", padx=15)

        # 控制按钮区（标题栏右侧）
        tk.Button(
            header, text="启动", width=6, bg=self.c_green, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._start
        ).pack(side="right", padx=3, pady=8)
        tk.Button(
            header, text="停止", width=6, bg=self.c_red, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._stop
        ).pack(side="right", padx=3, pady=8)
        tk.Button(
            header, text="浏览器", width=6, bg=self.c_gray, fg="white",
            font=("Microsoft YaHei", 9), command=self._open_browser
        ).pack(side="right", padx=3, pady=8)

        # Notebook 标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=5)

        self._build_overview_tab()
        self._build_positions_tab()
        self._build_strategies_tab()
        self._build_intervention_tab()

        # 底部日志区
        log_outer = tk.Frame(self.root, bg=self.c_bg)
        log_outer.pack(padx=8, pady=(0, 5), fill="x")
        tk.Label(
            log_outer, text="运行日志:", font=("Microsoft YaHei", 9, "bold"),
            bg=self.c_bg
        ).pack(anchor="w")
        log_frame = tk.Frame(log_outer, bg=self.c_bg)
        log_frame.pack(fill="x")
        self.txt_log = tk.Text(
            log_frame, height=7, width=110, state="disabled",
            font=("Consolas", 9), bg="#2c3e50", fg="#ecf0f1",
            wrap="word", relief="flat"
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=log_scroll.set)
        self.txt_log.pack(side="left", fill="x", expand=True)
        log_scroll.pack(side="right", fill="y")

    # --------------------------------------------------------------------
    # Tab 1: 概览
    # --------------------------------------------------------------------

    def _build_overview_tab(self):
        tab = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(tab, text="  概览  ")

        # 账户状态面板
        acct_frame = tk.LabelFrame(
            tab, text="账户状态", font=("Microsoft YaHei", 10, "bold"),
            bg=self.c_bg, padx=10, pady=8
        )
        acct_frame.pack(pady=8, padx=12, fill="x")

        r1 = tk.Frame(acct_frame, bg=self.c_bg)
        r1.pack(fill="x", pady=2)
        tk.Label(r1, text="运行状态:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=0, sticky="w")
        self.lbl_status = tk.Label(r1, text="已停止", fg=self.c_red, font=("Microsoft YaHei", 10, "bold"), bg=self.c_bg)
        self.lbl_status.grid(row=0, column=1, sticky="w", padx=(5, 25))
        tk.Label(r1, text="当前权益:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=2, sticky="w")
        self.lbl_equity = tk.Label(r1, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_equity.grid(row=0, column=3, sticky="w", padx=(5, 25))
        tk.Label(r1, text="今日盈亏:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=4, sticky="w")
        self.lbl_pnl = tk.Label(r1, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_pnl.grid(row=0, column=5, sticky="w", padx=5)

        r2 = tk.Frame(acct_frame, bg=self.c_bg)
        r2.pack(fill="x", pady=2)
        tk.Label(r2, text="持仓数量:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=0, sticky="w")
        self.lbl_positions = tk.Label(r2, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_positions.grid(row=0, column=1, sticky="w", padx=(5, 25))
        tk.Label(r2, text="风控状态:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=2, sticky="w")
        self.lbl_risk = tk.Label(r2, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_risk.grid(row=0, column=3, sticky="w", padx=(5, 25))
        tk.Label(r2, text="仓位放大:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=4, sticky="w")
        self.lbl_boost = tk.Label(r2, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_boost.grid(row=0, column=5, sticky="w", padx=5)

        r3 = tk.Frame(acct_frame, bg=self.c_bg)
        r3.pack(fill="x", pady=2)
        tk.Label(r3, text="回撤幅度:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=0, sticky="w")
        self.lbl_drawdown = tk.Label(r3, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_drawdown.grid(row=0, column=1, sticky="w", padx=(5, 25))
        tk.Label(r3, text="总盈亏:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=2, sticky="w")
        self.lbl_total_pnl = tk.Label(r3, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_total_pnl.grid(row=0, column=3, sticky="w", padx=(5, 25))
        tk.Label(r3, text="WS连接:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=4, sticky="w")
        self.lbl_ws = tk.Label(r3, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_ws.grid(row=0, column=5, sticky="w", padx=5)

        # 资金利用率面板
        util_frame = tk.LabelFrame(
            tab, text="资金利用率", font=("Microsoft YaHei", 10, "bold"),
            bg=self.c_bg, padx=10, pady=8
        )
        util_frame.pack(pady=5, padx=12, fill="x")

        ur1 = tk.Frame(util_frame, bg=self.c_bg)
        ur1.pack(fill="x", pady=2)
        tk.Label(ur1, text="当前利用率:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=0, sticky="w")
        self.lbl_util_rate = tk.Label(ur1, text="--", font=("Microsoft YaHei", 11, "bold"), bg=self.c_bg)
        self.lbl_util_rate.grid(row=0, column=1, sticky="w", padx=(5, 30))
        tk.Label(ur1, text="目标:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=2, sticky="w")
        self.lbl_util_target = tk.Label(ur1, text="85%", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_util_target.grid(row=0, column=3, sticky="w", padx=(5, 30))
        tk.Label(ur1, text="状态:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=4, sticky="w")
        self.lbl_util_status = tk.Label(ur1, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_util_status.grid(row=0, column=5, sticky="w", padx=5)

        ur2 = tk.Frame(util_frame, bg=self.c_bg)
        ur2.pack(fill="x", pady=2)
        tk.Label(ur2, text="已用保证金:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=0, sticky="w")
        self.lbl_util_used = tk.Label(ur2, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_util_used.grid(row=0, column=1, sticky="w", padx=(5, 30))
        tk.Label(ur2, text="可用余额:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=2, sticky="w")
        self.lbl_util_avail = tk.Label(ur2, text="--", font=("Microsoft YaHei", 10), bg=self.c_bg)
        self.lbl_util_avail.grid(row=0, column=3, sticky="w", padx=5)

        self.util_progress = ttk.Progressbar(util_frame, length=820, mode="determinate",
                                              maximum=100, value=0)
        self.util_progress.pack(pady=5)

        util_btn_frame = tk.Frame(util_frame, bg=self.c_bg)
        util_btn_frame.pack(pady=3)
        tk.Button(
            util_btn_frame, text="重置资金利用率", width=16, bg=self.c_purple, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._reset_utilization
        ).pack(side="left", padx=5)
        tk.Button(
            util_btn_frame, text="刷新", width=8, bg=self.c_gray, fg="white",
            font=("Microsoft YaHei", 9), command=self._refresh_all
        ).pack(side="left", padx=5)

    # --------------------------------------------------------------------
    # Tab 2: 持仓管理
    # --------------------------------------------------------------------

    def _build_positions_tab(self):
        tab = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(tab, text="  持仓管理  ")

        btn_frame = tk.Frame(tab, bg=self.c_bg)
        btn_frame.pack(pady=5, padx=12, fill="x")
        tk.Button(
            btn_frame, text="刷新持仓", width=12, bg=self.c_blue, fg="white",
            font=("Microsoft YaHei", 9), command=self._refresh_positions
        ).pack(side="left", padx=3)
        tk.Button(
            btn_frame, text="平仓选中", width=12, bg=self.c_orange, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._close_selected_position
        ).pack(side="left", padx=3)

        # 持仓表格
        tree_frame = tk.Frame(tab, bg=self.c_bg)
        tree_frame.pack(pady=5, padx=12, fill="both", expand=True)

        self.pos_tree = ttk.Treeview(
            tree_frame,
            columns=("symbol", "side", "qty", "avg", "mark", "pnl", "pnl_pct", "margin", "lev"),
            show="headings", height=12
        )
        cols = [
            ("symbol", "币种", 110),
            ("side", "方向", 60),
            ("qty", "数量", 90),
            ("avg", "开仓价", 90),
            ("mark", "标记价", 90),
            ("pnl", "未实现盈亏", 100),
            ("pnl_pct", "盈亏%", 70),
            ("margin", "保证金", 90),
            ("lev", "杠杆", 50),
        ]
        for cid, title, width in cols:
            self.pos_tree.heading(cid, text=title)
            self.pos_tree.column(cid, width=width, anchor="center")

        pos_scroll = ttk.Scrollbar(tree_frame, command=self.pos_tree.yview)
        self.pos_tree.configure(yscrollcommand=pos_scroll.set)
        self.pos_tree.pack(side="left", fill="both", expand=True)
        pos_scroll.pack(side="right", fill="y")

        # 持仓标签
        self.lbl_pos_count = tk.Label(tab, text="共 0 个持仓", font=("Microsoft YaHei", 9), bg=self.c_bg)
        self.lbl_pos_count.pack(pady=3)

    # --------------------------------------------------------------------
    # Tab 3: 策略管理
    # --------------------------------------------------------------------

    def _build_strategies_tab(self):
        tab = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(tab, text="  策略管理  ")

        btn_frame = tk.Frame(tab, bg=self.c_bg)
        btn_frame.pack(pady=5, padx=12, fill="x")
        tk.Button(
            btn_frame, text="刷新统计", width=12, bg=self.c_blue, fg="white",
            font=("Microsoft YaHei", 9), command=self._refresh_strategies
        ).pack(side="left", padx=3)
        tk.Button(
            btn_frame, text="启用/禁用切换", width=14, bg=self.c_orange, fg="white",
            font=("Microsoft YaHei", 9), command=self._toggle_selected_strategy
        ).pack(side="left", padx=3)

        tree_frame = tk.Frame(tab, bg=self.c_bg)
        tree_frame.pack(pady=5, padx=12, fill="both", expand=True)

        self.strat_tree = ttk.Treeview(
            tree_frame,
            columns=("strategy", "enabled", "open", "trades_24h", "wins", "losses",
                     "win_rate", "total_pnl", "avg_pnl", "fees"),
            show="headings", height=10
        )
        cols = [
            ("strategy", "策略", 90),
            ("enabled", "状态", 60),
            ("open", "开仓数", 60),
            ("trades_24h", "24h交易", 70),
            ("wins", "盈利", 50),
            ("losses", "亏损", 50),
            ("win_rate", "胜率%", 60),
            ("total_pnl", "总盈亏", 80),
            ("avg_pnl", "均盈亏", 70),
            ("fees", "手续费", 70),
        ]
        for cid, title, width in cols:
            self.strat_tree.heading(cid, text=title)
            self.strat_tree.column(cid, width=width, anchor="center")

        strat_scroll = ttk.Scrollbar(tree_frame, command=self.strat_tree.yview)
        self.strat_tree.configure(yscrollcommand=strat_scroll.set)
        self.strat_tree.pack(side="left", fill="both", expand=True)
        strat_scroll.pack(side="right", fill="y")

    # --------------------------------------------------------------------
    # Tab 4: 风控干预
    # --------------------------------------------------------------------

    def _build_intervention_tab(self):
        tab = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(tab, text="  风控干预  ")

        # 干预状态
        info_frame = tk.LabelFrame(
            tab, text="干预状态", font=("Microsoft YaHei", 10, "bold"),
            bg=self.c_bg, padx=10, pady=8
        )
        info_frame.pack(pady=8, padx=12, fill="x")

        tk.Label(info_frame, text="全局暂停:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=0, sticky="w")
        self.lbl_interv_paused = tk.Label(info_frame, text="--", font=("Microsoft YaHei", 10, "bold"), bg=self.c_bg)
        self.lbl_interv_paused.grid(row=0, column=1, sticky="w", padx=(5, 30))
        tk.Label(info_frame, text="风控暂停:", font=("Microsoft YaHei", 10), bg=self.c_bg).grid(row=0, column=2, sticky="w")
        self.lbl_interv_risk = tk.Label(info_frame, text="--", font=("Microsoft YaHei", 10, "bold"), bg=self.c_bg)
        self.lbl_interv_risk.grid(row=0, column=3, sticky="w", padx=5)

        # 常规操作
        normal_frame = tk.LabelFrame(
            tab, text="常规操作", font=("Microsoft YaHei", 10, "bold"),
            bg=self.c_bg, padx=10, pady=10
        )
        normal_frame.pack(pady=5, padx=12, fill="x")

        nf1 = tk.Frame(normal_frame, bg=self.c_bg)
        nf1.pack(pady=3)
        tk.Button(
            nf1, text="全局暂停", width=14, height=2, bg=self.c_orange, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._global_pause
        ).pack(side="left", padx=8)
        tk.Button(
            nf1, text="全局恢复", width=14, height=2, bg=self.c_green, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._global_resume
        ).pack(side="left", padx=8)
        tk.Button(
            nf1, text="重置风控暂停", width=14, height=2, bg=self.c_blue, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._reset_risk_pause
        ).pack(side="left", padx=8)

        nf2 = tk.Frame(normal_frame, bg=self.c_bg)
        nf2.pack(pady=3)
        tk.Button(
            nf2, text="撤销所有挂单", width=14, height=2, bg=self.c_gray, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._cancel_all_orders
        ).pack(side="left", padx=8)
        tk.Button(
            nf2, text="重置资金利用率", width=14, height=2, bg=self.c_purple, fg="white",
            font=("Microsoft YaHei", 9, "bold"), command=self._reset_utilization
        ).pack(side="left", padx=8)

        # 紧急操作
        emerg_frame = tk.LabelFrame(
            tab, text="紧急操作（需二次确认）", font=("Microsoft YaHei", 10, "bold"),
            bg=self.c_bg, padx=10, pady=10
        )
        emerg_frame.pack(pady=5, padx=12, fill="x")

        tk.Button(
            emerg_frame, text="紧急全平仓", width=20, height=3, bg=self.c_red, fg="white",
            font=("Microsoft YaHei", 11, "bold"), command=self._emergency_close_all
        ).pack(pady=5)

        tk.Label(
            emerg_frame, text="注意：紧急全平仓将立即平掉所有持仓，不可撤销",
            font=("Microsoft YaHei", 8), fg=self.c_red, bg=self.c_bg
        ).pack()

        # 干预历史
        hist_frame = tk.LabelFrame(
            tab, text="最近干预操作", font=("Microsoft YaHei", 10, "bold"),
            bg=self.c_bg, padx=10, pady=8
        )
        hist_frame.pack(pady=5, padx=12, fill="both", expand=True)

        self.hist_tree = ttk.Treeview(
            hist_frame, columns=("time", "action", "reason"), show="headings", height=5
        )
        self.hist_tree.heading("time", text="时间")
        self.hist_tree.heading("action", text="操作")
        self.hist_tree.heading("reason", text="原因")
        self.hist_tree.column("time", width=160, anchor="center")
        self.hist_tree.column("action", width=140, anchor="center")
        self.hist_tree.column("reason", width=400, anchor="w")
        self.hist_tree.pack(fill="both", expand=True)

    # ====================================================================
    # 日志
    # ====================================================================

    def _log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", f"{time.strftime('%H:%M:%S')} {msg}\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    # ====================================================================
    # 系统启停
    # ====================================================================

    def _start(self):
        if self.running:
            return
        self._log("正在启动...")
        self._kill_old()
        lock = os.path.join(BASE_DIR, "data", "app.lock")
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except PermissionError:
                pass

        self.core_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        self._log(f"核心系统 PID: {self.core_proc.pid}")

        hb = os.path.join(BASE_DIR, "data", "heartbeat.json")
        for _ in range(60):
            if os.path.exists(hb):
                break
            time.sleep(1)
        else:
            self._log("核心系统启动超时")
            self._stop()
            return
        time.sleep(3)

        self.dashboard_proc = subprocess.Popen(
            [sys.executable, "dashboard_api.py"],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._log(f"Dashboard PID: {self.dashboard_proc.pid}")

        import socket
        for _ in range(30):
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
        self.lbl_status.config(text="运行中", fg=self.c_green)
        self._log("系统已就绪")

        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        webbrowser.open(DASHBOARD_URL)

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

        self.lbl_status.config(text="已停止", fg=self.c_red)
        self._reset_all_labels()
        self._log("已停止")

    def _kill_old(self):
        my_pid = os.getpid()
        if os.name == "nt":
            os.system(f'taskkill /f /fi "PID ne {my_pid}" /im python.exe 2>nul')
            os.system(f'taskkill /f /fi "PID ne {my_pid}" /im pythonw.exe 2>nul')
        time.sleep(1)

    def _reset_all_labels(self):
        for lbl in [self.lbl_equity, self.lbl_pnl, self.lbl_positions, self.lbl_risk,
                    self.lbl_boost, self.lbl_drawdown, self.lbl_total_pnl, self.lbl_ws,
                    self.lbl_util_rate, self.lbl_util_status, self.lbl_util_used,
                    self.lbl_util_avail, self.lbl_interv_paused, self.lbl_interv_risk]:
            lbl.config(text="--", fg=self.c_gray)
        self.util_progress["value"] = 0

    # ====================================================================
    # 监控线程
    # ====================================================================

    def _monitor(self):
        while self.running:
            try:
                if self.core_proc and self.core_proc.poll() is not None:
                    self.root.after(0, lambda: self._log("核心进程已退出"))
                    self.root.after(0, self._stop)
                    break
                self._sync_overview()
                self._sync_risk_status()
                self._sync_utilization()
                self._sync_intervention()
            except Exception as e:
                self.root.after(0, lambda msg=str(e): self._log(f"同步错误: {msg}"))
            time.sleep(5)

    def _sync_overview(self):
        """从 /api/overview 同步账户概览"""
        data = api_get("/api/overview")
        if not data:
            return
        acct = data.get("account", {})
        trading = data.get("trading", {})
        sys_info = data.get("system", {})

        equity = acct.get("total_equity", 0)
        self.root.after(0, lambda e=equity: self.lbl_equity.config(text=f"{e:.2f} USDT"))

        open_pos = trading.get("open_positions", 0)
        self.root.after(0, lambda p=open_pos: self.lbl_positions.config(text=str(p)))

        # WS 连接状态从 /api/system_status 获取
        sys_data = api_get("/api/system_status")
        if sys_data:
            ws_ok = sys_data.get("ws_connected", False)
            ws_text = "已连接" if ws_ok else "未连接"
            ws_color = self.c_green if ws_ok else self.c_red
            self.root.after(0, lambda t=ws_text, c=ws_color: self.lbl_ws.config(text=t, fg=c))
            # 如果状态是 running，也更新状态标签
            if sys_data.get("status") == "running":
                self.root.after(0, lambda: self.lbl_status.config(text="运行中", fg=self.c_green))

    def _sync_risk_status(self):
        """从 /api/risk_status 同步风控状态"""
        data = api_get("/api/risk_status")
        if not data:
            return

        equity = data.get("current_equity", 0)
        daily_pnl = data.get("daily_pnl", 0)
        total_pnl = data.get("total_pnl", 0)
        drawdown = data.get("drawdown", 0)
        is_paused = data.get("is_paused", False)
        tier_triggered = data.get("tier_triggered", {})

        pnl_color = self.c_green if daily_pnl >= 0 else self.c_red
        self.root.after(0, lambda p=daily_pnl, c=pnl_color: self.lbl_pnl.config(text=f"{p:+.2f} USDT", fg=c))

        total_color = self.c_green if total_pnl >= 0 else self.c_red
        self.root.after(0, lambda p=total_pnl, c=total_color: self.lbl_total_pnl.config(text=f"{p:+.2f} USDT", fg=c))

        dd_color = self.c_red if drawdown > 0.15 else self.c_orange if drawdown > 0.08 else self.c_green
        self.root.after(0, lambda d=drawdown, c=dd_color: self.lbl_drawdown.config(text=f"{d:.1%}", fg=c))

        if is_paused:
            risk_text = "已暂停"
            risk_color = self.c_red
        elif any(tier_triggered.get(str(i), False) for i in [1, 2, 3]):
            active = [str(i) for i in [1, 2, 3] if tier_triggered.get(str(i), False)]
            risk_text = f"TIER{','.join(active)}"
            risk_color = self.c_orange
        else:
            risk_text = "正常"
            risk_color = self.c_green
        self.root.after(0, lambda t=risk_text, c=risk_color: self.lbl_risk.config(text=t, fg=c))

    def _sync_utilization(self):
        """从 /api/capital_utilization 同步资金利用率"""
        data = api_get("/api/capital_utilization")
        if not data:
            return

        rate = data.get("utilization_rate", 0)
        used = data.get("used_margin", 0)
        avail = data.get("available_balance", 0)
        target = data.get("target_utilization", 0.85)
        status = data.get("status", "normal")
        position_boost = data.get("position_boost", 1.0)

        status_map = {
            "low": ("低", self.c_red),
            "warming_up": ("预热中", self.c_orange),
            "normal": ("正常", self.c_green),
            "high": ("过高", self.c_red),
        }
        status_text, status_color = status_map.get(status, (status, self.c_gray))

        self.root.after(0, lambda: self._update_util_ui(
            rate, used, avail, target, status_text, status_color, position_boost
        ))

    def _update_util_ui(self, rate, used, avail, target, status_text, status_color, position_boost):
        rate_pct = rate * 100
        self.lbl_util_rate.config(text=f"{rate_pct:.1f}%", fg=status_color)
        self.lbl_util_target.config(text=f"{target * 100:.0f}%")
        self.lbl_util_status.config(text=status_text, fg=status_color)
        self.lbl_util_used.config(text=f"{used:.2f} USDT")
        self.lbl_util_avail.config(text=f"{avail:.2f} USDT")
        self.util_progress["value"] = min(100, rate_pct)

        if position_boost and position_boost > 1.0:
            self.lbl_boost.config(text=f"x{position_boost:.2f}", fg=self.c_orange)
        else:
            self.lbl_boost.config(text="x1.00", fg=self.c_gray)

    def _sync_intervention(self):
        """从 /api/intervention/status 同步干预状态"""
        data = api_get("/api/intervention/status")
        if not data:
            return
        paused = data.get("global_paused", False)
        pa_text = "是" if paused else "否"
        pa_color = self.c_red if paused else self.c_green
        self.root.after(0, lambda t=pa_text, c=pa_color: self.lbl_interv_paused.config(text=t, fg=c))

        # 风控暂停状态从 risk_status 同步（已在 _sync_risk_status 处理）
        risk_data = api_get("/api/risk_status")
        if risk_data:
            risk_paused = risk_data.get("is_paused", False)
            r_text = "是" if risk_paused else "否"
            r_color = self.c_red if risk_paused else self.c_green
            self.root.after(0, lambda t=r_text, c=r_color: self.lbl_interv_risk.config(text=t, fg=c))

    # ====================================================================
    # 持仓管理
    # ====================================================================

    def _refresh_positions(self):
        threading.Thread(target=self._fetch_positions, daemon=True).start()

    def _fetch_positions(self):
        data = api_get("/api/positions")
        if not data:
            self.root.after(0, lambda: self._log("获取持仓失败"))
            return
        positions = data.get("positions", [])

        def update_ui():
            for item in self.pos_tree.get_children():
                self.pos_tree.delete(item)
            for pos in positions:
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("pnl_percent", 0)
                pnl_str = f"{pnl:+.2f}"
                pnl_pct_str = f"{pnl_pct:+.1f}%"
                self.pos_tree.insert("", "end", values=(
                    pos.get("symbol", ""),
                    pos.get("side", ""),
                    f"{pos.get('quantity', 0):.4f}",
                    f"{pos.get('avg_cost', 0):.4f}",
                    f"{pos.get('mark_price', 0):.4f}",
                    pnl_str,
                    pnl_pct_str,
                    f"{pos.get('margin', 0):.2f}",
                    f"{pos.get('leverage', 1)}x",
                ))
            self.lbl_pos_count.config(text=f"共 {len(positions)} 个持仓")

        self.root.after(0, update_ui)
        self.root.after(0, lambda: self._log(f"持仓已刷新: {len(positions)} 个"))

    def _close_selected_position(self):
        sel = self.pos_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先在表格中选择要平仓的持仓")
            return
        item = self.pos_tree.item(sel[0])
        values = item["values"]
        symbol = str(values[0])
        side = str(values[1]).lower()
        qty = values[2]

        confirm = messagebox.askyesno(
            "确认平仓",
            f"确认平仓以下持仓？\n\n"
            f"  币种: {symbol}\n"
            f"  方向: {side}\n"
            f"  数量: {qty}\n\n"
            f"将提交限价/市价平仓订单。"
        )
        if not confirm:
            return

        self._log(f"正在平仓 {symbol} {side}...")
        threading.Thread(
            target=self._close_position_async,
            args=(symbol, side), daemon=True
        ).start()

    def _close_position_async(self, symbol, side):
        ok, result = api_post("/api/position/close", {"symbol": symbol, "pos_side": side})
        if ok and result.get("success"):
            self.root.after(0, lambda: self._log(
                f"平仓成功: {symbol} {side} 订单号={result.get('order_id', '')}"
            ))
            self.root.after(0, lambda: messagebox.showinfo("平仓成功", f"订单已提交\n订单号: {result.get('order_id', '')}"))
            self._fetch_positions()
        else:
            err = result.get("error", result.get("msg", "未知错误"))
            self.root.after(0, lambda: self._log(f"平仓失败: {err}"))
            self.root.after(0, lambda e=err: messagebox.showerror("平仓失败", e))

    # ====================================================================
    # 策略管理
    # ====================================================================

    def _refresh_strategies(self):
        threading.Thread(target=self._fetch_strategies, daemon=True).start()

    def _fetch_strategies(self):
        data = api_get("/api/strategies/stats")
        if not data:
            self.root.after(0, lambda: self._log("获取策略统计失败"))
            return
        strategies = data.get("strategies", {})

        def update_ui():
            for item in self.strat_tree.get_children():
                self.strat_tree.delete(item)
            for name, stats in strategies.items():
                enabled = stats.get("enabled", True)
                en_text = "启用" if enabled else "禁用"
                en_color_tag = enabled
                pnl = stats.get("total_pnl", 0)
                self.strat_tree.insert("", "end", values=(
                    name,
                    en_text,
                    stats.get("open_positions", 0),
                    stats.get("total_trades_24h", 0),
                    stats.get("wins", 0),
                    stats.get("losses", 0),
                    f"{stats.get('win_rate', 0):.1f}",
                    f"{pnl:+.4f}",
                    f"{stats.get('avg_pnl', 0):+.4f}",
                    f"{stats.get('total_fees', 0):.4f}",
                ))
        self.root.after(0, update_ui)
        self.root.after(0, lambda: self._log(f"策略统计已刷新: {len(strategies)} 个策略"))

    def _toggle_selected_strategy(self):
        sel = self.strat_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要切换的策略")
            return
        item = self.strat_tree.item(sel[0])
        strategy = str(item["values"][0])
        current_enabled = str(item["values"][1]) == "启用"
        new_enabled = not current_enabled

        action = "启用" if new_enabled else "禁用"
        confirm = messagebox.askyesno("确认", f"确认{action}策略 [{strategy}]？")
        if not confirm:
            return

        self._log(f"正在{action}策略 {strategy}...")
        threading.Thread(
            target=self._toggle_strategy_async,
            args=(strategy, new_enabled), daemon=True
        ).start()

    def _toggle_strategy_async(self, strategy, enabled):
        ok, result = api_post("/api/strategy/toggle", {"strategy": strategy, "enabled": enabled})
        if ok and result.get("success"):
            self.root.after(0, lambda: self._log(f"策略 {strategy} 已{'启用' if enabled else '禁用'}"))
            self._fetch_strategies()
        else:
            err = result.get("error", "未知错误")
            self.root.after(0, lambda: self._log(f"策略切换失败: {err}"))
            self.root.after(0, lambda e=err: messagebox.showerror("失败", e))

    # ====================================================================
    # 风控干预操作
    # ====================================================================

    def _global_pause(self):
        reason = messagebox.askquestion("全局暂停", "确认全局暂停？\n\n所有策略将停止开新仓。", type="yesno")
        if reason != "yes":
            return
        self._log("正在发送全局暂停请求...")
        threading.Thread(target=self._intervention_async,
                         args=("/api/intervention/global_pause",
                               {"reason": "客户端手动暂停"}), daemon=True).start()

    def _global_resume(self):
        if not messagebox.askyesno("确认", "确认全局恢复？"):
            return
        self._log("正在发送全局恢复请求...")
        threading.Thread(target=self._intervention_async,
                         args=("/api/intervention/global_resume", {}), daemon=True).start()

    def _reset_risk_pause(self):
        if not messagebox.askyesno("确认", "确认重置风控暂停状态？"):
            return
        self._log("正在发送重置风控暂停请求...")
        threading.Thread(target=self._intervention_async,
                         args=("/api/risk/reset_pause", {}), daemon=True).start()

    def _cancel_all_orders(self):
        if not messagebox.askyesno("确认", "确认撤销所有挂单？"):
            return
        self._log("正在发送撤单请求...")
        threading.Thread(target=self._intervention_async,
                         args=("/api/intervention/cancel_all_orders", {}), daemon=True).start()

    def _emergency_close_all(self):
        confirm1 = messagebox.askyesno(
            "紧急全平仓 - 第一次确认",
            "警告：将立即平掉所有持仓！\n\n此操作不可撤销。\n\n确认继续？"
        )
        if not confirm1:
            return
        confirm2 = messagebox.askyesno(
            "紧急全平仓 - 第二次确认",
            "再次确认：将立即平掉所有持仓！\n\n真的要继续吗？"
        )
        if not confirm2:
            return
        self._log("正在发送紧急全平仓请求...")
        threading.Thread(target=self._intervention_async,
                         args=("/api/intervention/emergency_close_all",
                               {"reason": "客户端紧急全平", "confirmed": True}),
                         daemon=True).start()

    def _intervention_async(self, path, payload):
        ok, result = api_post(path, payload)
        if ok and result.get("success"):
            msg = result.get("message", result.get("msg", "操作成功"))
            self.root.after(0, lambda m=msg: self._log(f"干预操作成功: {m}"))
        else:
            err = result.get("error", result.get("message", "未知错误"))
            self.root.after(0, lambda e=err: self._log(f"干预操作失败: {e}"))
            self.root.after(0, lambda e=err: messagebox.showerror("操作失败", e))

    # ====================================================================
    # 资金利用率重置
    # ====================================================================

    def _reset_utilization(self):
        if not self.running:
            messagebox.showwarning("提示", "系统未运行，无法重置")
            return
        confirm = messagebox.askyesno(
            "确认重置",
            "重置资金利用率将执行以下操作：\n\n"
            "  1. 清空利用率历史记录\n"
            "  2. 仓位放大系数重置为 x1.00\n"
            "  3. 恢复信号质量阈值到初始值\n"
            "  4. 重新进入 30 分钟预热期\n\n"
            "AdaptiveController 将在 ≤120 秒内响应。\n\n确认继续？"
        )
        if not confirm:
            return

        threading.Thread(target=self._reset_utilization_async, daemon=True).start()

    def _reset_utilization_async(self):
        ok, result = api_post("/api/capital_utilization/reset", {"source": "client_gui"})
        if ok and result.get("success"):
            msg = result.get("message", "重置信号已发送")
            self.root.after(0, lambda m=msg: self._log(f"资金利用率重置: {m}"))
            self.root.after(0, lambda: messagebox.showinfo("重置信号已发送",
                "AdaptiveController 将在下个检测周期（≤120秒）执行重置"))
        else:
            err = result.get("error", "未知错误")
            self.root.after(0, lambda: self._log(f"重置失败: {err}"))
            self.root.after(0, lambda e=err: messagebox.showerror("重置失败", e))

    # ====================================================================
    # 刷新全部
    # ====================================================================

    def _refresh_all(self):
        if not self.running:
            self._log("系统未运行")
            return
        self._log("正在刷新所有数据...")
        threading.Thread(target=self._refresh_all_async, daemon=True).start()

    def _refresh_all_async(self):
        self._sync_overview()
        self._sync_risk_status()
        self._sync_utilization()
        self._sync_intervention()
        self._fetch_positions()
        self._fetch_strategies()
        self.root.after(0, lambda: self._log("所有数据已刷新"))

    # ====================================================================
    # 初始状态检查 & 窗口关闭
    # ====================================================================

    def _refresh_status(self):
        import socket
        try:
            with socket.create_connection(("127.0.0.1", 8080), timeout=1):
                self.lbl_status.config(text="运行中(外部)", fg=self.c_orange)
                self.running = True
                self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
                self.monitor_thread.start()
                # 首次刷新
                self._refresh_all_async()
        except Exception:
            pass

    def _open_browser(self):
        webbrowser.open(DASHBOARD_URL)

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
