"""
OKX 量化交易系统主程序入口。
"""
import asyncio
import signal
import sys
import os
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from loguru import logger

# ── 消除 _readerthread UnicodeDecodeError 噪声 ──
# 第三方库（如 ccxt、OKX SDK）可能使用 subprocess.Popen(text=True, stdout=PIPE)
# 而不指定 encoding=，导致在中文 Windows(cp936) 上解码 UTF-8 输出时崩溃。
# 此 monkey-patch 为所有 subprocess.Popen 调用注入 errors='replace' 默认值。
import subprocess as _sp
_original_popen_init = _sp.Popen.__init__

def _patched_popen_init(self, args, bufsize=-1, executable=None,
                         stdin=None, stdout=None, stderr=None,
                         preexec_fn=None, close_fds=True, shell=False,
                         cwd=None, env=None, universal_newlines=None,
                         startupinfo=None, creationflags=0,
                         restore_signals=True, start_new_session=False,
                         pass_fds=(), *, encoding=None, errors=None,
                         text=None, group=None, extra_groups=None,
                         user=None, umask=-1, pipesize=-1, process_group=None):
    # 当 text=True 或 universal_newlines=True 但未显式指定 errors= 时，补齐 'replace'
    if (text or universal_newlines) and errors is None:
        errors = 'replace'
    # Python 3.10 不支持 process_group/group/extra_groups/user/umask/pipesize 参数
    import sys as _sys
    kwargs = dict(
        encoding=encoding, errors=errors, text=text)
    if _sys.version_info >= (3, 11):
        kwargs['process_group'] = process_group
    if _sys.version_info >= (3, 11):
        kwargs['group'] = group
        kwargs['extra_groups'] = extra_groups
        kwargs['user'] = user
        kwargs['umask'] = umask
        kwargs['pipesize'] = pipesize
    return _original_popen_init(self, args, bufsize, executable,
                                stdin, stdout, stderr, preexec_fn,
                                close_fds, shell, cwd, env,
                                universal_newlines, startupinfo,
                                creationflags, restore_signals,
                                start_new_session, pass_fds,
                                **kwargs)

_sp.Popen.__init__ = _patched_popen_init
# ── END monkey-patch ──

from core.scheduler import TradingScheduler
from configs.settings import load_config

_shutdown_event = asyncio.Event()

# 心跳线程停止信号（独立于asyncio事件循环，避免API hang导致心跳过期）
_heartbeat_stop = threading.Event()
_heartbeat_thread = None


def setup_logging():
    """配置日志文件持久化：按天分割，保留30天"""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 移除默认handler，配置文件+控制台双输出
    logger.remove()
    # 控制台：INFO级别，彩色，带事件ID
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <yellow>{extra[event_id]}</yellow> | <level>{message}</level>")
    # 文件：DEBUG级别，按天分割，保留30天，自动压缩
    logger.add(f"{log_dir}/trading_{{time:YYYY-MM-DD}}.log",
               level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[event_id]} | {name}:{function}:{line} - {message}",
               rotation="00:00",  # 每天0点分割
               retention="30 days",  # 保留30天
               compression="zip",  # 自动压缩历史日志
               encoding="utf-8",
               enqueue=True)  # 异步写入避免阻塞
    # 错误日志单独文件
    logger.add(f"{log_dir}/error_{{time:YYYY-MM-DD}}.log",
               level="ERROR",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[event_id]} | {name}:{function}:{line} - {message}",
               rotation="00:00",
               retention="90 days",
               compression="zip",
               encoding="utf-8",
               enqueue=True)

    # P22: 启用全链路事件ID体系
    from core.event_id import configure_event_id_logging
    configure_event_id_logging()

    logger.info("Logging configured: logs/ (daily rotation, 30d retention, event IDs enabled)")


def backup_database():
    """备份数据库到 backups/ 目录，按日期命名。

    使用 SQLite 在线备份 API（src.backup(dest)）而非 shutil.copy2，确保在
    WAL 模式下也能得到一致性快照；备份前执行 PRAGMA quick_check，校验失败则
    拒绝备份，防止用损坏数据覆盖可恢复的历史备份。
    """
    try:
        db_path = "./data/trading.db"
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return

        # 1. 完整性校验（quick_check 足够检测结构损坏，速度快）
        check_conn = sqlite3.connect(db_path)
        try:
            check_result = check_conn.execute("PRAGMA quick_check").fetchone()
        finally:
            check_conn.close()
        if not check_result or check_result[0] != "ok":
            logger.error(f"Database integrity check failed, skip backup: {check_result}")
            return

        # 2. 在线备份（WAL 安全，不阻塞主库写入）
        backup_dir = "./backups"
        os.makedirs(backup_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}/trading_{date_str}.db"

        src = sqlite3.connect(db_path)
        try:
            dest = sqlite3.connect(backup_path)
            try:
                src.backup(dest)
            finally:
                dest.close()
        finally:
            src.close()
        logger.info(f"Database backed up to {backup_path}")

        # 3. 清理超过30天的备份
        cutoff = datetime.now().timestamp() - 30 * 86400
        for fname in os.listdir(backup_dir):
            fpath = os.path.join(backup_dir, fname)
            if os.path.isfile(fpath) and fname.startswith("trading_"):
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    logger.debug(f"Removed old backup: {fname}")
    except Exception as e:
        logger.error(f"Database backup failed: {e}")


def signal_handler(signum, frame):
    logger.info("Received termination signal, initiating graceful shutdown...")
    _shutdown_event.set()


def _heartbeat_writer(interval: int = 10):
    """独立心跳线程：每 interval 秒写一次 data/heartbeat.json
    用 daemon 线程而非 asyncio task，确保即使 asyncio 事件循环卡住
    （如长时间 OKX API 调用）心跳仍能更新，避免 watchdog 误判进程卡死
    """
    heartbeat_path = os.path.join("data", "heartbeat.json")
    os.makedirs("data", exist_ok=True)
    start_time = datetime.now()
    pid = os.getpid()

    while not _heartbeat_stop.is_set():
        try:
            now = datetime.now()
            uptime = (now - start_time).total_seconds()
            # 获取当前活跃线程数（粗略反映系统负载）
            try:
                thread_count = threading.active_count()
            except Exception as e:
                logger.debug(f"Failed to get thread count: {e}")
                thread_count = -1

            heartbeat = {
                "pid": pid,
                "start_time": start_time.isoformat(),
                "last_update": now.isoformat(),
                "uptime_seconds": round(uptime, 1),
                "thread_count": thread_count,
                "status": "running"
            }
            # 原子写入：先写临时文件再 rename，避免 watchdog 读到半截 JSON
            tmp_path = heartbeat_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(heartbeat, f, ensure_ascii=False)
            os.replace(tmp_path, heartbeat_path)
        except Exception as e:
            # 心跳写入失败不能让线程退出
            logger.debug(f"Heartbeat write failed: {e}")
        # 用 wait 而不是 sleep，收到停止信号能立即响应
        _heartbeat_stop.wait(timeout=interval)


def start_heartbeat():
    """启动心跳线程"""
    global _heartbeat_thread
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_writer,
        name="heartbeat",
        daemon=True  # daemon=True：主进程退出时自动结束，不阻塞 shutdown
    )
    _heartbeat_thread.start()
    logger.info("Heartbeat thread started (interval=10s, independent of asyncio loop)")


def stop_heartbeat():
    """停止心跳线程并清理心跳文件"""
    _heartbeat_stop.set()
    if _heartbeat_thread and _heartbeat_thread.is_alive():
        _heartbeat_thread.join(timeout=3)
    # 标记为已停止，让 watchdog 知道是主动退出而非崩溃
    try:
        heartbeat_path = os.path.join("data", "heartbeat.json")
        if os.path.exists(heartbeat_path):
            with open(heartbeat_path, "r+", encoding="utf-8") as f:
                hb = json.load(f)
                hb["status"] = "stopped"
                hb["last_update"] = datetime.now().isoformat()
                f.seek(0)
                f.truncate()
                json.dump(hb, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to update heartbeat file on shutdown: {e}")


async def shutdown(scheduler: TradingScheduler):
    logger.info("Starting graceful shutdown...")

    await scheduler.shutdown()

    # 停止心跳线程
    stop_heartbeat()

    # 关闭前再备份一次数据库
    backup_database()

    logger.info("Graceful shutdown complete")
    sys.exit(0)


async def daily_maintenance():
    """每日维护任务：凌晨2点备份数据库"""
    while True:
        try:
            now = datetime.now()
            # 计算到明天凌晨2点（用timedelta避免月末越界）
            tomorrow_2am = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if tomorrow_2am <= now:
                tomorrow_2am = (now + timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
            wait_seconds = (tomorrow_2am - now).total_seconds()
            await asyncio.sleep(wait_seconds)

            logger.info("Starting daily maintenance: backing up database...")
            backup_database()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Daily maintenance error: {e}")
            await asyncio.sleep(3600)


async def _run_connectivity_check(diag):
    """运行网络连通性检查（后台任务，不阻塞启动）"""
    try:
        result = await diag.check_connectivity()
        if result.fail_count == 0:
            logger.info(f"Connectivity check:\n{result.report()}")
        else:
            logger.warning(f"Connectivity check:\n{result.report()}")
        # 检查账户
        acct = await diag.check_account()
        if acct.get("checked"):
            usdt = acct.get("balance_usdt")
            if usdt is not None:
                level = "OK" if usdt >= 100 else ("WARN" if usdt >= 20 else "FAIL")
                logger.info(f"Account balance: {usdt:.2f} USDT [{level}]")
    except Exception as e:
        logger.debug(f"Connectivity check skipped: {e}")


async def main():
    from utils.single_instance import SingleInstance
    instance = SingleInstance()
    # 重试获取互斥锁：watchdog 重启时旧进程的互斥锁可能尚未释放
    max_retries, retry_delay = 10, 2
    for attempt in range(1, max_retries + 1):
        if instance.acquire():
            break
        if attempt == max_retries:
            sys.exit(1)
        time.sleep(retry_delay)

    try:
        # signal 只能在主线程中使用（嵌入模式下交易系统运行在子线程）
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            pass

        # 配置日志持久化
        setup_logging()

        # P0 修复：安装全局异常处理器（sys.excepthook + asyncio loop 双兜底）
        # 必须在事件循环运行中调用，才能拿到 running loop 设置 asyncio 异常处理器
        try:
            from core.exception_handler import install_global_exception_handler
            install_global_exception_handler()
        except Exception as e:
            logger.warning(f"Failed to install global exception handler: {e}")

        # 启动时备份一次数据库
        backup_database()

        logger.info("Starting OKX Radical Quant Trading System...")

        # 启动独立心跳线程（必须在scheduler.start之前，确保watchdog不会误判启动期）
        start_heartbeat()

        config = load_config()

        # 模块9：密钥失效自检与轮换提醒（启动即告警，避免密钥失效时静默下单失败）
        try:
            from core.api_key_manager import check_key_health, check_key_rotation_reminder
            key_health = check_key_health()
            if not key_health["healthy"]:
                logger.warning(f"API key health: {key_health['issues']}")
            rotation = check_key_rotation_reminder()
            if rotation["need_rotation"]:
                logger.warning(
                    f"API key rotation overdue: {rotation['age_days']} days "
                    f"(threshold {rotation['max_age_days']} days)"
                )
        except Exception as e:
            logger.warning(f"Key health check skipped: {e}")

        # ============================================================
        # 系统自诊断：启动时全面检查关键组件
        # ============================================================
        from utils.system_diagnostics import SystemDiagnostics, DiagLevel
        diag = SystemDiagnostics(config)
        result = await diag.run_all()
        logger.info(f"System self-diagnosis:\n{result.report()}")

        if result.fail_count > 0:
            logger.warning(f"Self-diagnosis: {result.fail_count} FAIL(s) detected, system may be degraded")

        # 网络连通性检查（异步，不阻塞启动）
        asyncio.create_task(_run_connectivity_check(diag))
        # ============================================================

        scheduler = TradingScheduler(config)

        # 启动每日维护任务
        maintenance_task = asyncio.create_task(daily_maintenance())

        await scheduler.start()

        await _shutdown_event.wait()

        maintenance_task.cancel()
        await shutdown(scheduler)
    finally:
        instance.release()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)