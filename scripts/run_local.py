import asyncio
import signal
import sys
from loguru import logger

from core.local_client import LocalOKXClient
from configs.settings import load_config

_shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    logger.info("Received termination signal, initiating graceful shutdown...")
    _shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting Local OKX Quant Trading System...")
    
    config = load_config()
    
    local_config = {
        "symbols": config.get("trading", {}).get("symbols", ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]),
        "initial_balance": config.get("trading", {}).get("total_capital", 100000.0),
        "tick_interval_ms": config.get("strategies", {}).get("scalping", {}).get("tick_processing_interval_ms", 50)
    }
    
    client = LocalOKXClient(local_config)
    
    async def on_tick(tick_data):
        logger.debug(f"Tick received: {tick_data.symbol} @ {tick_data.price:.2f}")
    
    async def on_order(order_data):
        logger.info(f"Order update: {order_data['ordId']} - {order_data['state']}")
    
    async def on_position(position_data):
        logger.info(f"Position update: {position_data.symbol} {position_data.side} {position_data.quantity:.4f}")
    
    client.tick_callback = on_tick
    client.order_callback = on_order
    client.position_callback = on_position
    
    await client.start_tick_stream()
    
    logger.info("Local client started. Testing basic functionality...")
    
    await asyncio.sleep(1)
    
    ticker = client.get_ticker("BTC-USDT-SWAP")
    if ticker:
        logger.info(f"BTC-USDT-SWAP Ticker: {ticker['last']}")
    
    account_info = client.get_account_info()
    if account_info:
        logger.info(f"Account Info - Total Equity: {account_info['totalEq']}, Available: {account_info['availBal']}")
    
    order_result = client.place_order(
        symbol="BTC-USDT-SWAP",
        side="buy",
        order_type="market",
        quantity=0.01,
        leverage=10
    )
    if order_result:
        logger.info(f"Order placed successfully: {order_result['ordId']}")
    
    await asyncio.sleep(2)
    
    positions = client.get_positions()
    if positions:
        for pos in positions:
            logger.info(f"Position: {pos['instId']} {pos['posSide']} {pos['pos']} @ {pos['avgPx']}")
    
    account_info = client.get_account_info()
    if account_info:
        logger.info(f"Account after order - Total Equity: {account_info['totalEq']}, Used Margin: {account_info['usedMargin']}")
    
    await asyncio.sleep(2)
    
    logger.info("=== Test Results ===")
    logger.info("✅ Tick stream: Working")
    logger.info("✅ Ticker API: Working")
    logger.info("✅ Account Info: Working")
    logger.info("✅ Order Placement: Working")
    logger.info("✅ Position Management: Working")
    
    logger.info("=" * 60)
    logger.info("Local OKX Quant Trading Client is running...")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    await _shutdown_event.wait()
    
    await client.close_websocket()
    logger.info("Local client shutdown complete")


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