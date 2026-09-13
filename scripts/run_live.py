import asyncio
import signal
import sys
import os
from loguru import logger

from core.okx_client import OKXClient
from core.scheduler import TradingScheduler
from configs.settings import load_config
from main import setup_logging

_shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    logger.info("Received termination signal, initiating graceful shutdown...")
    _shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    setup_logging()

    logger.info("=" * 70)
    logger.info("  WARNING: LIVE TRADING MODE ENABLED")
    logger.info("  REAL FUNDS WILL BE USED - USE WITH CAUTION")
    logger.info("=" * 70)

    config = load_config()

    api_key = os.getenv("OKX_API_KEY", "")
    secret_key = os.getenv("OKX_SECRET_KEY", "")
    passphrase = os.getenv("OKX_PASSPHRASE", "")

    if not api_key or not secret_key or not passphrase:
        logger.error("ERROR: API credentials not configured!")
        logger.error("Please update the .env file with your OKX API credentials")
        logger.error("OKX_API_KEY=your_api_key")
        logger.error("OKX_SECRET_KEY=your_secret_key")
        logger.error("OKX_PASSPHRASE=your_passphrase")
        sys.exit(1)

    if api_key == "your_real_api_key_here":
        logger.error("ERROR: API credentials are still set to default values!")
        logger.error("Please update the .env file with your actual OKX API credentials")
        sys.exit(1)

    logger.info("Verifying API connection...")

    okx_client = OKXClient(config)

    try:
        account_info = okx_client.get_account_info() if hasattr(okx_client, 'get_account_info') else okx_client._make_request("GET", "/api/v5/account/balance")
        if account_info:
            if isinstance(account_info, dict) and account_info.get('totalEq'):
                logger.info("[OK] API Connection Successful")
                logger.info(f"   Total Equity: {account_info.get('totalEq', 'N/A')} USDT")
                logger.info(f"   Available Balance: {account_info.get('availBal', 'N/A')} USDT")
                logger.info(f"   Used Margin: {account_info.get('usedMargin', 'N/A')} USDT")
            elif isinstance(account_info, dict) and account_info.get('data') and len(account_info['data']) > 0:
                data = account_info['data'][0]
                logger.info("[OK] API Connection Successful")
                logger.info(f"   Total Equity: {data.get('totalEq', 'N/A')} USDT")
                logger.info(f"   Available Balance: {data.get('availBal', 'N/A')} USDT")
                logger.info(f"   Used Margin: {data.get('usedMargin', 'N/A')} USDT")
            else:
                logger.info("[OK] API Connection Successful (raw response format)")
        else:
            logger.error("[FAIL] Failed to connect to OKX API")
            logger.error("Please check your API credentials and network connection")
            sys.exit(1)
    except Exception as e:
        logger.error(f"[FAIL] API Connection Failed: {e}")
        logger.error("Please check:")
        logger.error("  1. API Key, Secret Key, Passphrase are correct")
        logger.error("  2. API permissions are enabled (Trade, Read)")
        logger.error("  3. Network connection is working")
        logger.error("  4. IP whitelist (if enabled) includes your IP")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("LIVE TRADING SYSTEM INITIALIZATION")
    logger.info("=" * 70)

    logger.info(f"Mode: {'Live Trading' if not config['okx'].get('is_testnet', False) else 'Testnet'}")
    logger.info(f"API: {config['okx']['rest_url']}")
    logger.info(f"Total Capital: {config['trading']['total_capital']} USDT")
    logger.info(f"Trading Capital: {config['trading']['total_capital'] * config['trading']['trading_capital_ratio']:.2f} USDT")
    logger.info(f"Max Drawdown: {config['trading']['max_drawdown'] * 100:.1f}%")
    logger.info(f"Daily Max Loss: {config['trading']['daily_max_loss'] * 100:.1f}%")
    logger.info(f"Architecture: Service Layer (SignalProcessor)")
    logger.info("=" * 70)

    scheduler = TradingScheduler(config)

    logger.info("Starting Trading Scheduler with Service Layer...")

    try:
        await scheduler.start()
    except Exception as e:
        logger.error(f"Trading scheduler failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        await _shutdown_event.wait()
    finally:
        await scheduler.shutdown()
        logger.info("Shutdown complete")


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
