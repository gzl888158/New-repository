"""
进程守护脚本：监控交易系统进程，崩溃时自动重启
用法：
    python watchdog.py              # 启动守护，监控main.py
    python watchdog.py --no-restart # 仅监控不重启
"""
import os
import sys
import time
import json
import signal
import subprocess
from datetime import datetime, timedelta
from loguru import logger


class TradingWatchdog:
    """交易系统进程守护"""

    def __init__(self, restart_on_crash: bool = True, max_restarts: int = 10):
        self.restart_on_crash = restart_on_crash
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.restart_window = 3600  # 1小时内最多重启max_restarts次
        self.restart_times: list = []
        self.process: subprocess.Popen = None
        self.start_time = None

        # 项目根目录（watchdog.py所在目录），用于子进程cwd和日志绝对路径
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        # 配置日志
        logger.remove()
        log_dir = os.path.join(self.project_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        logger.add(f"{log_dir}/watchdog_{{time:YYYY-MM-DD}}.log",
                   level="DEBUG",
                   rotation="00:00",
                   retention="30 days",
                   compression="zip",
                   encoding="utf-8",
                   enqueue=True)

    def start_trading_system(self) -> subprocess.Popen:
        """启动交易系统"""
        venv_python = os.path.join(self.project_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            venv_python = sys.executable

        main_script = os.path.join(self.project_dir, "main.py")
        log_dir = os.path.join(self.project_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 用datetime生成日期文件名，重定向子进程stdout/stderr到日志文件
        # 避免DEVNULL静默吞掉启动期错误（如import失败、配置错误）
        date_str = datetime.now().strftime("%Y%m%d")
        stdout_log = os.path.join(log_dir, f"main_stdout_{date_str}.log")
        stderr_log = os.path.join(log_dir, f"main_stderr_{date_str}.log")

        logger.info(f"Starting trading system: {venv_python} main.py")
        logger.info(f"  cwd     -> {self.project_dir}")
        logger.info(f"  stdout  -> {stdout_log}")
        logger.info(f"  stderr  -> {stderr_log}")

        # 用当前环境变量传递给子进程，确保输出编码一致
        # PYTHONIOENCODING=utf-8:replace 让 subprocess 内部 decode 错误时用 � 替代而非崩溃
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8:replace"
        env["PYTHONUTF8"] = "1"  # PEP 540: 强制 UTF-8 模式

        # 打开日志文件（二进制append模式），避免 subprocess module 在 Windows 上
        # 创建 _readerthread 线程时因编码问题产生 UnicodeDecodeError（如 0xb5 字节）。
        # 子进程 logger 输出已是 UTF-8，二进制写入原样保留。
        stdout_fh = open(stdout_log, "ab")
        stderr_fh = open(stderr_log, "ab")
        # 写入一条标记行（UTF-8 BOM-like），方便区分重启边界
        restart_marker = f"\n{'='*60}\n{datetime.now().isoformat()} RESTART\n{'='*60}\n".encode("utf-8")
        stdout_fh.write(restart_marker)
        stderr_fh.write(restart_marker)
        stdout_fh.flush()
        stderr_fh.flush()

        try:
            # 用新进程启动，独立于watchdog
            # Windows下CREATE_NEW_PROCESS_GROUP让我们后续能用CTRL_BREAK_EVENT优雅退出
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            self.process = subprocess.Popen(
                [venv_python, main_script],
                cwd=self.project_dir,
                stdout=stdout_fh,
                stderr=stderr_fh,
                env=env,
                creationflags=creationflags
            )
        finally:
            # 子进程已继承fd，父进程关闭自己的句柄避免泄漏
            stdout_fh.close()
            stderr_fh.close()

        self.start_time = datetime.now()
        logger.info(f"Trading system started, PID={self.process.pid}")

        # 探测早期失败：5秒内退出说明启动失败（如import失败、配置错误）
        # 原本DEVNULL会吞掉这些错误，现在重定向到日志文件后可以定位
        try:
            self.process.wait(timeout=5)
            # wait返回说明进程已退出
            exit_code = self.process.returncode
            logger.error(f"Trading system exited within 5s! exit_code={exit_code}")
            logger.error(f"  Check stderr log for details: {stderr_log}")
            # 读取stderr日志最后几行帮助定位启动失败原因
            try:
                with open(stderr_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                if lines:
                    # 过滤 _readerthread 噪声（第三方库 subprocess 编码问题，无害）
                    lines = [l for l in lines if "_readerthread" not in l and "UnicodeDecodeError" not in l]
                    if lines:
                        tail = "".join(lines[-20:])
                        logger.error(f"Last 20 lines of stderr:\n{tail}")
            except Exception as e:
                logger.error(f"Failed to read stderr log: {e}")
        except subprocess.TimeoutExpired:
            # 进程仍在运行，正常
            pass

        return self.process

    def is_process_alive(self) -> bool:
        """检查进程是否存活"""
        if self.process is None:
            return False
        return self.process.poll() is None

    def is_heartbeat_alive(self) -> bool:
        """检查心跳是否健康：读取 data/heartbeat.json 的 last_update 时间戳
        超过60秒认为不健康（进程可能卡死/死锁/网络hang但未退出）

        启动后给予120秒grace period，让系统有时间写入首次心跳，
        避免误判启动期心跳缺失导致立即重启
        """
        # 启动后grace period：避免误判启动期心跳缺失
        if self.start_time and (datetime.now() - self.start_time).total_seconds() < 120:
            return True

        heartbeat_path = os.path.join(self.project_dir, "data", "heartbeat.json")
        try:
            if not os.path.exists(heartbeat_path):
                logger.warning(f"Heartbeat file not found: {heartbeat_path}")
                return False
            with open(heartbeat_path, "r", encoding="utf-8") as f:
                hb = json.load(f)
            last_update_str = hb.get("last_update")
            if not last_update_str:
                logger.warning("Heartbeat file missing 'last_update' field")
                return False
            # 兼容ISO字符串和数字时间戳两种格式
            try:
                last_update = datetime.fromisoformat(str(last_update_str))
            except (ValueError, TypeError):
                try:
                    last_update = datetime.fromtimestamp(float(last_update_str))
                except (ValueError, TypeError):
                    logger.warning(f"Cannot parse last_update: {last_update_str}")
                    return False
            age = (datetime.now() - last_update).total_seconds()
            if age > 60:
                logger.warning(f"Heartbeat expired: last_update={last_update}, age={age:.0f}s > 60s")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking heartbeat: {e}")
            return False

    def should_restart(self) -> bool:
        """检查是否应该重启（避免无限重启循环）"""
        if not self.restart_on_crash:
            return False

        # 清理超过1小时的重启记录
        now = time.time()
        self.restart_times = [t for t in self.restart_times if now - t < self.restart_window]

        if len(self.restart_times) >= self.max_restarts:
            logger.critical(f"Max restarts ({self.max_restarts}) in {self.restart_window}s window reached, "
                            f"giving up to avoid crash loop")
            return False

        return True

    def stop(self):
        """停止交易系统"""
        if self.process and self.is_process_alive():
            logger.info("Stopping trading system...")
            try:
                # 优先触发优雅退出，让main.py的signal handler清理未平仓订单
                # - Windows: CTRL_BREAK_EVENT发送到进程组（需配合CREATE_NEW_PROCESS_GROUP）
                #   process.terminate()等价于TerminateProcess，不会触发handler，可能丢失订单
                # - Unix: SIGTERM
                if os.name == 'nt':
                    try:
                        logger.info(f"Sending CTRL_BREAK_EVENT to PID={self.process.pid}")
                        os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                    except Exception as e:
                        logger.warning(f"CTRL_BREAK_EVENT failed, fallback to terminate(): {e}")
                        self.process.terminate()
                else:
                    self.process.terminate()

                # 等待10秒优雅关闭
                try:
                    self.process.wait(timeout=10)
                    logger.info("Trading system stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Trading system did not stop in 10s, killing")
                    self.process.kill()
                    self.process.wait()
                    logger.info("Trading system killed")
            except Exception as e:
                logger.error(f"Error stopping trading system: {e}")

    def _cleanup_lock_file(self):
        """清理锁文件，防止僵尸锁阻止重启"""
        lock_path = os.path.join(self.project_dir, "data", "app.lock")
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
                logger.info(f"Cleaned up stale lock file: {lock_path}")
        except Exception as e:
            logger.warning(f"Failed to clean lock file: {e}")

    def _dump_crash_diagnostics(self):
        """崩溃时输出最近的 stderr 日志，帮助诊断崩溃原因"""
        log_dir = os.path.join(self.project_dir, "logs")
        date_str = datetime.now().strftime("%Y%m%d")
        stderr_log = os.path.join(log_dir, f"main_stderr_{date_str}.log")
        try:
            if os.path.exists(stderr_log):
                with open(stderr_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                if lines:
                    # 过滤 _readerthread 噪声
                    lines = [l for l in lines if "_readerthread" not in l and "UnicodeDecodeError" not in l]
                    if lines:
                        tail = "".join(lines[-30:])
                        logger.error(f"CRASH DIAGNOSTICS — last 30 lines of stderr:\n=====\n{tail}=====")
                    else:
                        logger.error("CRASH DIAGNOSTICS — stderr contains only _readerthread noise (no actionable errors)")
                else:
                    logger.error("CRASH DIAGNOSTICS — stderr log is empty")
            else:
                logger.error(f"Stderr log not found: {stderr_log}")
        except Exception as e:
            logger.error(f"Failed to read crash diagnostics: {e}")

    def _write_status(self, phase: str, extra: dict = None):
        """写入启动状态文件供外部监控"""
        status_path = os.path.join(self.project_dir, "data", "startup_status.json")
        try:
            status = {
                "phase": phase,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pid": self.process.pid if self.process else None,
                "restart_count": len(self.restart_times),
            }
            if extra:
                status.update(extra)
            os.makedirs(os.path.dirname(status_path), exist_ok=True)
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 静默失败，不影响核心逻辑

    def _check_memory(self, pid: int = None) -> float:
        """检查进程内存使用（MB），返回 None 表示无法获取"""
        try:
            import psutil
            p = psutil.Process(pid or self.process.pid)
            mem_mb = p.memory_info().rss / 1024 / 1024
            return mem_mb
        except ImportError:
            return None
        except Exception:
            return None

    def _wait_old_process_gone(self, timeout: int = 15):
        """等待旧进程完全退出并确保 Windows 内核互斥锁被释放。
        在 stop() 后调用，确保 start_trading_system() 不会因互斥锁残留而失败。"""
        deadline = time.time() + timeout
        old_pid = self.process.pid if self.process else None

        # 1) 等待进程真正退出
        while self.process and self.is_process_alive():
            if time.time() > deadline:
                logger.warning(f"Old process PID={old_pid} did not die in {timeout}s, force killing")
                try:
                    self.process.kill()
                except Exception:
                    pass
                break
            time.sleep(0.5)

        # 2) 额外等待 Windows 内核释放命名互斥锁
        #    CreateMutex 创建的命名对象在最后一个句柄关闭后才销毁，需要一点时间。
        mutex_cleanup_delay = 3
        logger.info(f"Waiting {mutex_cleanup_delay}s for Windows kernel mutex cleanup...")
        time.sleep(mutex_cleanup_delay)

    def run(self):
        """主监控循环"""
        logger.info("=" * 60)
        logger.info("Trading Watchdog started")
        logger.info(f"  Restart on crash: {self.restart_on_crash}")
        logger.info(f"  Max restarts: {self.max_restarts}/hour")
        logger.info(f"  Heartbeat timeout: 60s (with 120s startup grace period)")
        logger.info("=" * 60)

        # 首次启动
        self.start_trading_system()
        self._write_status("running")

        check_interval = 30  # 每30秒检查一次
        while True:
            try:
                time.sleep(check_interval)

                if not self.is_process_alive():
                    exit_code = self.process.returncode
                    uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                    logger.error(f"Trading system CRASHED! exit_code={exit_code}, uptime={uptime}")

                    # 崩溃诊断 + 清理
                    self._dump_crash_diagnostics()
                    self._cleanup_lock_file()
                    self._write_status("crashed", {"exit_code": exit_code, "uptime_seconds": int(uptime.total_seconds())})

                    if self.should_restart():
                        # 重启前等待，避免快速崩溃循环
                        wait_time = min(60, 10 + len(self.restart_times) * 10)
                        logger.info(f"Waiting {wait_time}s before restart...")
                        time.sleep(wait_time)

                        logger.error(f"Restarting trading system (attempt {len(self.restart_times) + 1}/{self.max_restarts})")
                        self.restart_times.append(time.time())
                        # 等待旧进程互斥锁释放
                        self._wait_old_process_gone()
                        self._write_status("restarting")
                        self.start_trading_system()
                    else:
                        logger.critical("Watchdog giving up, exiting")
                        self._write_status("dead", {"reason": "max_restarts_exceeded"})
                        break
                elif not self.is_heartbeat_alive():
                    # 进程存活但心跳过期：可能死锁/asyncio卡死/网络hang，强制重启
                    logger.warning(f"Trading system heartbeat EXPIRED, PID={self.process.pid} alive but unhealthy")
                    logger.error(f"Force restarting due to heartbeat failure (attempt {len(self.restart_times) + 1}/{self.max_restarts})")

                    self._dump_crash_diagnostics()
                    self._write_status("heartbeat_lost")

                    if self.should_restart():
                        wait_time = min(60, 10 + len(self.restart_times) * 10)
                        logger.info(f"Waiting {wait_time}s before restart...")
                        time.sleep(wait_time)

                        self.restart_times.append(time.time())
                        # 先stop旧进程...
                        self.stop()
                        self._cleanup_lock_file()
                        # 等待旧进程完全退出 + Windows 内核清理互斥锁
                        self._wait_old_process_gone()
                        logger.info("Old process confirmed dead, starting new instance...")
                        self._write_status("restarting")
                        self.start_trading_system()
                    else:
                        logger.critical("Watchdog giving up, exiting")
                        self._write_status("dead", {"reason": "max_restarts_exceeded"})
                        break
                else:
                    # 进程健康，记录心跳
                    uptime = datetime.now() - self.start_time
                    if int(uptime.total_seconds()) % 300 < check_interval:  # 每5分钟记录一次
                        mem_mb = self._check_memory()
                        mem_str = f", memory={mem_mb:.0f}MB" if mem_mb else ""
                        logger.info(f"Heartbeat: trading system running, PID={self.process.pid}, uptime={uptime}{mem_str}")
                        self._write_status("running", {"uptime_seconds": int(uptime.total_seconds())})

            except KeyboardInterrupt:
                logger.info("Watchdog interrupted by user")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(10)


if __name__ == "__main__":
    no_restart = "--no-restart" in sys.argv
    watchdog = TradingWatchdog(restart_on_crash=not no_restart)
    watchdog.run()
