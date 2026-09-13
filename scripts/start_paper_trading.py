"""
P22-5: 生产级模拟盘/沙盒交易启动脚本
====================================
提供完整的模拟盘交易环境，用于实盘验证前的策略测试。

运行方式：
    python scripts/start_paper_trading.py
    python scripts/start_paper_trading.py --capital 5000
    python scripts/start_paper_trading.py --mode sandbox --strategies grid,trend
    python scripts/start_paper_trading.py --compliance-only

特性：
- 沙盒模式：完全隔离的虚拟交易环境，使用OKX实时行情
- 测试网模式：连接OKX测试网API进行模拟交易
- 延迟监控：自动暂停开仓当行情延迟超标
- 限价优先：优先使用限价单减少滑点
- 合规检查：定期生产级红线合规检查
- 状态持久化：支持暂停恢复，不丢失交易记录
- 绩效报告：完整的资金曲线、胜率、盈亏比统计
"""

import argparse
import asyncio
import os
import sys
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_paper_logging():
    """配置模拟盘专用日志"""
    log_dir = Path("./logs/paper_trading")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 移除默认handler
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <yellow>{extra[event_id]}</yellow> | <level>{message}</level>"
    )
    
    # 文件日志
    date_str = datetime.now().strftime("%Y%m%d")
    logger.add(
        log_dir / f"paper_trading_{date_str}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[event_id]} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )
    
    # 启用事件ID
    try:
        from core.event_id import configure_event_id_logging
        configure_event_id_logging()
    except Exception:
        pass
    
    logger.info("=" * 60)
    logger.info("OKX量化交易系统 - 模拟盘/沙盒交易环境")
    logger.info("=" * 60)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="OKX量化交易系统 - 模拟盘/沙盒交易环境",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/start_paper_trading.py                          # 默认配置启动
  python scripts/start_paper_trading.py --capital 5000            # 5000 USDT初始资金
  python scripts/start_paper_trading.py --strategies grid,trend   # 仅运行网格和趋势策略
  python scripts/start_paper_trading.py --mode sandbox            # 沙盒模式(默认)
  python scripts/start_paper_trading.py --compliance-only         # 仅运行合规检查
        """
    )
    
    parser.add_argument(
        "--capital", type=float, default=None,
        help="初始资金 (USDT), 默认从config.yaml读取"
    )
    parser.add_argument(
        "--mode", type=str, default="sandbox", choices=["sandbox", "testnet"],
        help="运行模式: sandbox(沙盒) 或 testnet(测试网), 默认sandbox"
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="逗号分隔的策略列表, 如: grid,trend,scalping"
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的交易对列表, 如: BTC,ETH,SOL"
    )
    parser.add_argument(
        "--leverage", type=int, default=None,
        help="最大杠杆倍数"
    )
    parser.add_argument(
        "--no-limit-priority", action="store_true",
        help="禁用限价优先策略"
    )
    parser.add_argument(
        "--no-latency-pause", action="store_true",
        help="禁用延迟暂停开仓"
    )
    parser.add_argument(
        "--compliance-only", action="store_true",
        help="仅运行合规检查后退出"
    )
    parser.add_argument(
        "--report-interval", type=int, default=3600,
        help="绩效报告间隔(秒), 默认3600"
    )
    
    return parser.parse_args()


def build_paper_config(args, base_config: dict) -> dict:
    """根据命令行参数构建模拟盘配置"""
    pt_config = base_config.get("paper_trading", {}).copy()
    
    if args.capital is not None:
        pt_config["initial_capital"] = args.capital
    
    if args.mode:
        pt_config["mode"] = args.mode
    
    if args.strategies:
        pt_config["strategies"] = [s.strip() for s in args.strategies.split(",")]
    
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        pt_config["symbols"] = [
            f"{s}-USDT-SWAP" if not s.endswith("-SWAP") else s
            for s in symbols
        ]
    
    if args.leverage is not None:
        pt_config["max_leverage"] = args.leverage
    
    if args.no_limit_priority:
        pt_config["limit_order_priority"] = False
    
    if args.no_latency_pause:
        pt_config["latency_pause_enabled"] = False
    
    pt_config["enabled"] = True
    
    base_config["paper_trading"] = pt_config
    return base_config


async def run_compliance_check_only(config: dict):
    """仅运行合规检查"""
    logger.info("Running production compliance check...")
    
    from core.paper_trading_bridge import PaperTradingBridge
    
    bridge = PaperTradingBridge(config)
    result = bridge.run_compliance_check()
    
    print("\n" + "=" * 60)
    print("  生产级红线合规检查报告")
    print("=" * 60)
    print(f"  状态: {result['status']}")
    print(f"  检查时间: {result['checked_at']}")
    print(f"  性能指标:")
    perf = result['performance']
    print(f"    ROI: {perf.get('roi_pct', 0):.2f}%")
    print(f"    胜率: {perf.get('win_rate_pct', 0):.1f}%")
    print(f"    盈亏比: {perf.get('profit_factor', 0):.2f}")
    print(f"    最大回撤: {perf.get('max_drawdown_pct', 0):.1f}%")
    print(f"    总交易: {perf.get('total_trades', 0)}")
    print(f"  延迟: {result['latency'].get('avg_latency_ms', 0):.0f}ms")
    
    if result['issues']:
        print(f"\n  ❌ 问题 ({len(result['issues'])}):")
        for issue in result['issues']:
            print(f"     - {issue}")
    
    if result['warnings']:
        print(f"\n  ⚠️  警告 ({len(result['warnings'])}):")
        for warning in result['warnings']:
            print(f"     - {warning}")
    
    if not result['issues'] and not result['warnings']:
        print(f"\n  ✅ 所有检查通过，无问题发现")
    
    print("=" * 60)


async def print_performance_report(bridge, interval: int):
    """定期打印绩效报告"""
    while True:
        try:
            await asyncio.sleep(interval)
            perf = bridge.get_performance_summary()
            
            print("\n" + "-" * 50)
            print(f"  📊 模拟盘绩效报告 [{datetime.now().strftime('%H:%M:%S')}]")
            print("-" * 50)
            print(f"  ROI: {perf.get('roi_pct', 0):+.2f}%")
            print(f"  胜率: {perf.get('win_rate_pct', 0):.1f}% "
                  f"({perf.get('winning_trades', 0)}W / {perf.get('losing_trades', 0)}L)")
            print(f"  盈亏比: {perf.get('profit_factor', 0):.2f}")
            print(f"  总交易: {perf.get('total_trades_count', 0)}")
            print(f"  最大回撤: {perf.get('max_drawdown_pct', 0):.1f}%")
            print(f"  K线数: {perf.get('bridge_bar_count', 0)}")
            print(f"  延迟: {perf.get('latency', {}).get('avg_latency_ms', 0):.0f}ms")
            print(f"  开仓允许: {'✅' if perf.get('opening_allowed', True) else '❌ 暂停'}")
            print("-" * 50)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Report error: {e}")


async def main():
    args = parse_args()
    setup_paper_logging()
    
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模式: {args.mode}")
    
    # 加载配置
    from configs.settings import load_config
    config = load_config()
    
    # 构建模拟盘配置
    config = build_paper_config(args, config)
    pt_config = config.get("paper_trading", {})
    
    logger.info(f"初始资金: {pt_config.get('initial_capital', 10000)} USDT")
    logger.info(f"策略: {pt_config.get('strategies', [])}")
    logger.info(f"交易对: {len(pt_config.get('symbols', []))} 个")
    logger.info(f"最大杠杆: {pt_config.get('max_leverage', 5)}x")
    logger.info(f"限价优先: {pt_config.get('limit_order_priority', True)}")
    logger.info(f"延迟暂停: {pt_config.get('latency_pause_enabled', True)}")
    
    # 仅合规检查模式
    if args.compliance_only:
        await run_compliance_check_only(config)
        return
    
    # 创建模拟盘桥接器
    from core.paper_trading_bridge import PaperTradingBridge, get_paper_trading_bridge
    
    bridge = get_paper_trading_bridge(config)
    
    # 启动桥接器
    if not await bridge.start():
        logger.error("Failed to start paper trading bridge")
        return
    
    # 设置信号处理
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal...")
        shutdown_event.set()
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        pass
    
    # 启动绩效报告
    report_task = asyncio.create_task(
        print_performance_report(bridge, args.report_interval)
    )
    
    # 启动合规检查定时器
    async def compliance_loop():
        interval = pt_config.get("compliance_check_interval_sec", 3600)
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)
                result = bridge.run_compliance_check()
                if not result['passed']:
                    logger.warning(
                        f"P22-8: Compliance check FAILED - {len(result['issues'])} issues: "
                        f"{'; '.join(result['issues'])}"
                    )
                for w in result.get('warnings', []):
                    logger.info(f"P22-8: Compliance warning: {w}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Compliance check error: {e}")
    
    if pt_config.get("compliance_check_enabled", True):
        compliance_task = asyncio.create_task(compliance_loop())
    else:
        compliance_task = None
    
    # 模拟行情注入（从WebSocket行情流获取）
    logger.info("Paper trading bridge running. Waiting for market data...")
    logger.info("Connect WebSocket feeds to inject market data via bridge.on_bar() / bridge.on_tick()")
    logger.info("Press Ctrl+C to stop")
    
    # 等待关闭信号
    await shutdown_event.wait()
    
    # 清理
    logger.info("Shutting down paper trading...")
    report_task.cancel()
    if compliance_task:
        compliance_task.cancel()
    
    # 最终合规检查
    final_check = bridge.run_compliance_check()
    logger.info(f"Final compliance check: {final_check['status']}")
    
    # 停止桥接器
    summary = await bridge.stop()
    
    logger.info("=" * 60)
    logger.info("  模拟盘交易结束")
    logger.info(f"  最终ROI: {summary.get('roi_pct', 0):+.2f}%")
    logger.info(f"  总交易: {summary.get('total_trades_count', 0)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)