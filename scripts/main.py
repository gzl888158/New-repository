import asyncio
import signal
import sys
import os
import json
import shutil
import threading
import time
from datetime import datetime, timedelta
from loguru import logger

from core.scheduler import TradingScheduler
from configs.settings import load_config

_shutdown_event = asyncio.Event()

_heartbeat_stop = threading.Event()
_heartbeat_thread = None


def setup_logging():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(f"{log_dir}/trading_{{time:YYYY-MM-DD}}.log",
               level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
               rotation="00:00",
               retention="30 days",
               compression="zip",
               encoding="utf-8",
               enqueue=True)
    logger.add(f"{log_dir}/error_{{time:YYYY-MM-DD}}.log",
               level="ERROR",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
               rotation="00:00",
               retention="90 days",
               compression="zip",
               encoding="utf-8",
               enqueue=True)

    logger.info("Logging configured: logs/ (daily rotation, 30d retention)")


def backup_database():
    try:
        db_path = "./data/trading.db"
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return

        backup_dir = "./backups"
        os.makedirs(backup_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}/trading_{date_str}.db"
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")

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
    heartbeat_path = os.path.join("data", "heartbeat.json")
    os.makedirs("data", exist_ok=True)
    start_time = datetime.now()
    pid = os.getpid()

    while not _heartbeat_stop.is_set():
        try:
            now = datetime.now()
            uptime = (now - start_time).total_seconds()
            try:
                thread_count = threading.active_count()
            except Exception:
                thread_count = -1

            heartbeat = {
                "pid": pid,
                "start_time": start_time.isoformat(),
                "last_update": now.isoformat(),
                "uptime_seconds": round(uptime, 1),
                "thread_count": thread_count,
                "status": "running"
            }
            tmp_path = heartbeat_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(heartbeat, f, ensure_ascii=False)
            os.replace(tmp_path, heartbeat_path)
        except Exception as e:
            logger.debug(f"Heartbeat write failed: {e}")
        _heartbeat_stop.wait(timeout=interval)


def start_heartbeat():
    global _heartbeat_thread
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_writer,
        name="heartbeat",
        daemon=True
    )
    _heartbeat_thread.start()
    logger.info("Heartbeat thread started (interval=10s, independent of asyncio loop)")


def stop_heartbeat():
    _heartbeat_stop.set()
    if _heartbeat_thread and _heartbeat_thread.is_alive():
        _heartbeat_thread.join(timeout=3)
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
    except Exception:
        pass


async def shutdown(scheduler: TradingScheduler):
    logger.info("Starting graceful shutdown...")

    await scheduler.shutdown()

    stop_heartbeat()

    backup_database()

    logger.info("Graceful shutdown complete")
    sys.exit(0)


async def daily_maintenance():
    while True:
        try:
            now = datetime.now()
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


async def main():
    from utils.single_instance import SingleInstance
    instance = SingleInstance()
    if not instance.acquire():
        sys.exit(1)

    try:
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            pass

        setup_logging()

        backup_database()

        logger.info("Starting OKX Radical Quant Trading System...")
        logger.info("Architecture: Service Layer (SignalProcessor, TradingSchedulerService)")

        start_heartbeat()

        config = load_config()

        scheduler = TradingScheduler(config)

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
        traceback.print_exc()
        sys.exit(1)
