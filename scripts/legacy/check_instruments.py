"""Check instrument info and try XRP order"""
from dotenv import load_dotenv; load_dotenv()
from configs.settings import load_config
from core.okx_client import OKXClient
import json

config = load_config()
client = OKXClient(config)

# Check XRP-USDT-SWAP instrument info
info = client.get_instrument_info("XRP-USDT-SWAP")
print("=== XRP-USDT-SWAP Instrument Info ===")
if info:
    print(f"  ctVal: {info.get('ctVal')} (contract value)")
    print(f"  ctMult: {info.get('ctMult')} (contract multiplier)")
    print(f"  lotSz: {info.get('lotSz')} (lot size)")
    print(f"  minSz: {info.get('minSz')} (min size)")
    print(f"  tickSz: {info.get('tickSz')} (tick size)")
    print(f"  ctType: {info.get('ctType')} (contract type)")
    print(f"  uly: {info.get('uly')} (underlying)")
    print(f"  lever: {info.get('lever')} (max leverage)")
else:
    print("  No info!")

# Also check DOGE for comparison
info2 = client.get_instrument_info("DOGE-USDT-SWAP")
print("\n=== DOGE-USDT-SWAP Instrument Info ===")
if info2:
    print(f"  ctVal: {info2.get('ctVal')} (contract value)")
    print(f"  ctMult: {info2.get('ctMult')} (contract multiplier)")
    print(f"  lotSz: {info2.get('lotSz')} (lot size)")
    print(f"  minSz: {info2.get('minSz')} (min size)")
