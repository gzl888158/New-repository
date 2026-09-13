"""Test margin requirement for XRP-USDT-SWAP"""
from dotenv import load_dotenv; load_dotenv()
from configs.settings import load_config
from core.okx_client import OKXClient
import json, time

config = load_config()
client = OKXClient(config)

# Get current XRP price
ticker = client.get_ticker("XRP-USDT-SWAP")
if ticker:
    last_price = float(ticker.get("last", 0))
    print(f"XRP last price: {last_price}")

# Test order at near-market price to see real margin requirement
# Use very small quantity
print("\n=== Test XRP buy at near-market, sz=0.01 ===")
data = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "XRP-USDT-SWAP",
    "tdMode": "isolated",
    "side": "buy",
    "ordType": "limit",
    "sz": "0.01",
    "px": str(last_price * 0.95),  # 5% below market
    "lever": "3",
    "posSide": "long"
}))
print(f"Result: {data}")
if data and data.get("code") == "0":
    ord_id = data["data"][0]["ordId"]
    print(f"\nOrder placed: {ord_id}")
    # Cancel immediately
    time.sleep(0.5)
    cancel = client._make_request("POST", "/api/v5/trade/cancel-order", json.dumps({
        "instId": "XRP-USDT-SWAP", "ordId": ord_id
    }))
    print(f"Cancel result: {cancel.get('code')}")

# Also try with sz=0.1 at near-market
print("\n=== Test XRP buy at near-market, sz=0.1 ===")
data2 = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "XRP-USDT-SWAP",
    "tdMode": "isolated",
    "side": "buy",
    "ordType": "limit",
    "sz": "0.1",
    "px": str(last_price * 0.95),
    "lever": "3",
    "posSide": "long"
}))
print(f"Result: {data2}")
if data2 and data2.get("code") == "0":
    ord_id2 = data2["data"][0]["ordId"]
    time.sleep(0.5)
    cancel2 = client._make_request("POST", "/api/v5/trade/cancel-order", json.dumps({
        "instId": "XRP-USDT-SWAP", "ordId": ord_id2
    }))
    print(f"Cancel result: {cancel2.get('code')}")
