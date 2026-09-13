"""Test minimum order sizes with isolated margin"""
from dotenv import load_dotenv; load_dotenv()
from configs.settings import load_config
from core.okx_client import OKXClient
import json

config = load_config()
client = OKXClient(config)

# Test minimum XRP order (lotSz=0.01)
print("=== Test XRP-USDT-SWAP min order (0.01 contracts) ===")
data = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "XRP-USDT-SWAP",
    "tdMode": "isolated",
    "side": "buy",
    "ordType": "limit",
    "sz": "0.01",
    "px": "0.50",  # far below market to avoid fill
    "lever": "3",
    "posSide": "long"
}))
print(f"Result: {data}")

# Test with sz=0.1
if data and data.get("code") == "0":
    print("\n=== Test XRP-USDT-SWAP 0.1 contracts ===")
    data2 = client._make_request("POST", "/api/v5/trade/order", json.dumps({
        "instId": "XRP-USDT-SWAP",
        "tdMode": "isolated",
        "side": "buy",
        "ordType": "limit",
        "sz": "0.1",
        "px": "0.50",
        "lever": "3",
        "posSide": "long"
    }))
    print(f"Result: {data2}")

    # Test with sz=0.5
    if data2 and data2.get("code") == "0":
        print("\n=== Test XRP-USDT-SWAP 0.5 contracts ===")
        data3 = client._make_request("POST", "/api/v5/trade/order", json.dumps({
            "instId": "XRP-USDT-SWAP",
            "tdMode": "isolated",
            "side": "buy",
            "ordType": "limit",
            "sz": "0.5",
            "px": "0.50",
            "lever": "3",
            "posSide": "long"
        }))
        print(f"Result: {data3}")
