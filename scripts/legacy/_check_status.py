import sys, os, json
sys.path.insert(0, '.')
from core.okx_client import OKXClient
from configs.settings import load_config

config = load_config()
client = OKXClient(config)

# Get account info
try:
    info = client.get_account_info()
    print('=== ACCOUNT INFO ===')
    if info:
        keys = ['totalEq', 'availBal', 'notionalUsd', 'upl', 'mgnRatio', 'adjEq']
        for k in keys:
            print(f"  {k}: {info.get(k, 'N/A')}")
    else:
        print('No account data')
except Exception as e:
    print(f'Account error: {e}')

# Get positions
try:
    positions = client.get_positions()
    print(f'\n=== POSITIONS ({len(positions) if positions else 0}) ===')
    if positions:
        for p in positions:
            instId = p.get('instId', '?')
            posSide = p.get('posSide', '?')
            pos = p.get('pos', '?')
            avgPx = p.get('avgPx', '?')
            upl = p.get('upl', '?')
            margin = p.get('margin', '?')
            print(f"  {instId} {posSide} qty={pos} avgPx={avgPx} upl={upl} margin={margin}")
    else:
        print('No positions')
except Exception as e:
    print(f'Positions error: {e}')