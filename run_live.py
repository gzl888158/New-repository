"""
实盘交易模式入口，使用真实 API 资金运行交易系统。

所有生命周期管理（预检、单实例锁、Kill Switch 检查、优雅启停）均由
`live.LiveTradingRunner` 负责，本文件仅作为薄入口。
"""
import asyncio
import json
import sys

from loguru import logger

from configs.settings import load_config
from main import setup_logging
from live import LiveTradingRunner


async def main():
    # 配置日志持久化
    setup_logging()

    # 加载配置（不做拷贝，由 LiveTradingRunner 内部深拷贝）
    config = load_config()

    # 创建运行器并执行完整生命周期
    runner = LiveTradingRunner(config)
    report = await runner.run()

    # 打印最终运行报告
    status = report.get("status", "unknown")
    logger.info("=" * 70)
    logger.info(f"LIVE TRADING SESSION ENDED: status={status}")
    logger.info(f"  duration: {report.get('duration_seconds', 0):.1f}s")
    if report.get("error"):
        logger.error(f"  error: {report['error']}")
    failed = report.get("preflight", {}).get("failed_names", [])
    if failed:
        logger.error(f"  failed preflight checks: {failed}")
    logger.info("=" * 70)

    # 持久化运行报告到 logs 目录
    try:
        import os
        from datetime import datetime
        os.makedirs("./logs", exist_ok=True)
        report_path = f"./logs/live_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Session report saved to {report_path}")
    except Exception as e:
        logger.warning(f"Failed to save session report: {e}")

    # 非成功状态以非零退出码退出，便于 watchdog / systemd 感知
    if status not in ("shutdown_normal",):
        sys.exit(1)


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
