import sys
import os
import asyncio
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from configs.settings import load_config
from core.okx_client import OKXClient
from data.redis_cache import RedisCache
from services.market_data_service import MarketDataService, DataQualityChecker
from core.models import TickData

config = load_config()

print("=" * 70)
print("数据源接入系统验证")
print("=" * 70)

# 1. 验证配置加载
print("\n1. 配置检查")
md_config = config.get("market_data", {})
print(f"  max_data_delay_seconds: {md_config.get('max_data_delay_seconds', 'N/A')}")
print(f"  max_price_change_pct: {md_config.get('max_price_change_pct', 'N/A')}")
print(f"  rest_fallback_interval_seconds: {md_config.get('rest_fallback_interval_seconds', 'N/A')}")
print(f"  quality_monitor_interval_seconds: {md_config.get('quality_monitor_interval_seconds', 'N/A')}")

# 2. 验证数据质量检查器
print("\n2. 数据质量校验")
quality_checker = DataQualityChecker(config)
tick = TickData(
    symbol="BTC-USDT-SWAP",
    price=65000.0,
    volume=1000.0,
    bid_price=64999.0,
    bid_volume=1.0,
    ask_price=65001.0,
    ask_volume=1.0,
    timestamp=datetime.now()
)
report = quality_checker.check_tick(tick)
print(f"  正常tick质量分数: {report['quality_score']}")
print(f"  问题: {report['issues']}")

bad_tick = TickData(
    symbol="BTC-USDT-SWAP",
    price=0.0,
    volume=1000.0,
    bid_price=64999.0,
    bid_volume=1.0,
    ask_price=65001.0,
    ask_volume=1.0,
    timestamp=datetime.now()
)
bad_report = quality_checker.check_tick(bad_tick)
print(f"  异常tick质量分数: {bad_report['quality_score']}")
print(f"  问题: {bad_report['issues']}")

# 3. 验证Redis缓存新鲜度
print("\n3. Redis缓存新鲜度")
redis_cache = RedisCache(config)
redis_cache.set_tick(tick, ttl_seconds=60)
fresh = redis_cache.is_tick_fresh("BTC-USDT-SWAP", max_age_seconds=30)
print(f"  tick是否新鲜: {fresh}")
retrieved = redis_cache.get_tick("BTC-USDT-SWAP", max_age_seconds=30)
print(f"  读取tick价格: {retrieved.price if retrieved else 'N/A'}")

# 4. 验证WebSocket状态接口
print("\n4. WebSocket连接状态接口")
okx_client = OKXClient(config)
status = okx_client.get_ws_status()
print(f"  public_connected: {status['public_connected']}")
print(f"  private_connected: {status['private_connected']}")
print(f"  public_subscribed_channels: {status['public_subscribed_channels']}")
print(f"  private_subscribed_channels: {status['private_subscribed_channels']}")

# 5. 验证REST API可用性
print("\n5. REST API可用性")
try:
    ticker = okx_client.get_ticker("BTC-USDT-SWAP")
    if ticker:
        print(f"  BTC最新价: {ticker.get('last', 'N/A')}")
        print(f"  买一/卖一: {ticker.get('bidPx', 'N/A')} / {ticker.get('askPx', 'N/A')}")
    else:
        print("  获取ticker失败")
except Exception as e:
    print(f"  REST API异常: {e}")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
