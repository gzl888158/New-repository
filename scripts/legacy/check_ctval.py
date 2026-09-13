"""Check ctVal for all trading symbols"""
from dotenv import load_dotenv; load_dotenv()
from configs.settings import load_config
from core.okx_client import OKXClient

config = load_config()
client = OKXClient(config)

symbols = ["XRP-USDT-SWAP", "ADA-USDT-SWAP", "DOGE-USDT-SWAP", "ARB-USDT-SWAP"]
for sym in symbols:
    info = client.get_instrument_info(sym)
    if info:
        ctVal = info.get('ctVal', 'N/A')
        lotSz = info.get('lotSz', 'N/A')
        minSz = info.get('minSz', 'N/A')
        print(f"{sym}: ctVal={ctVal}, lotSz={lotSz}, minSz={minSz}")
