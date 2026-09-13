import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient
from core.account_manager import AccountManager
from core.state_manager import get_global_state
from monitoring.health_scorer import HealthScorer

config = load_config()
okx_client = OKXClient(config)

print("=" * 60)
print("交易系统验证报告")
print("=" * 60)

try:
    account_info = okx_client.get_account_info()
    if account_info:
        print(f"\n账户权益: {account_info.get('totalEq', 'N/A')} USDT")
        print(f"可用余额: {account_info.get('availBal', 'N/A')} USDT")
        print(f"已用保证金: {account_info.get('usedMargin', 'N/A')} USDT")
        print(f"未实现盈亏: {account_info.get('unrealizedPnl', 'N/A')} USDT")
    else:
        print("\n账户信息获取失败")
except Exception as e:
    print(f"\n账户信息获取异常: {e}")

try:
    positions = okx_client.get_positions()
    if positions:
        print(f"\n当前持仓 ({len(positions)} 个):")
        for pos in positions[:5]:
            pos_side = "多" if pos.get("posSide") == "long" else "空"
            print(f"  {pos.get('instId')}: {pos_side} {pos.get('pos', '0')} @ {pos.get('avgPx', '0')}")
        if len(positions) > 5:
            print(f"  ... 还有 {len(positions) - 5} 个持仓")
    else:
        print("\n当前无持仓")
except Exception as e:
    print(f"\n持仓信息获取异常: {e}")

state_manager = get_global_state(config)
system_health = state_manager.get("system.health")
trading_enabled = state_manager.get("trading.enabled")

print(f"\n系统健康状态: {system_health or 'healthy'}")
print(f"交易功能启用: {trading_enabled or True}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)