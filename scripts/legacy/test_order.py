"""Test OKX order placement without posSide"""
import os, sys, json
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

from configs.settings import load_config
from core.okx_client import OKXClient

config = load_config()
client = OKXClient(config)

# Check account config
acct = client.get_account_info()
if acct:
    print(f"acctLv={acct.get('acctLv')}, totalEq={acct.get('totalEq')}, mgnRatio={acct.get('mgnRatio')}")
    for d in acct.get("details", []):
        if d.get("ccy") == "USDT":
            print(f"USDT: eq={d.get('eq')}, availBal={d.get('availBal')}, cashBal={d.get('cashBal')}, frozenBal={d.get('frozenBal')}, availEq={d.get('availEq')}")

# Test 1: Try isolated without posSide
print("\n--- Test isolated without posSide ---")
data = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "DOGE-USDT-SWAP",
    "tdMode": "isolated",
    "side": "buy",
    "ordType": "limit",
    "sz": "1",
    "px": "0.05",
    "lever": "3"
}))
print(f"Result: {data}")

# Test 2: Try cross without posSide
print("\n--- Test cross without posSide ---")
data = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "DOGE-USDT-SWAP",
    "tdMode": "cross",
    "side": "buy", 
    "ordType": "limit",
    "sz": "1",
    "px": "0.05",
    "lever": "3"
}))
print(f"Result: {data}")

# Test 3: Try with posSide=long isolated
print("\n--- Test isolated with posSide=long ---")
data = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "DOGE-USDT-SWAP",
    "tdMode": "isolated",
    "side": "buy",
    "ordType": "limit",
    "sz": "1",
    "px": "0.05",
    "lever": "3",
    "posSide": "long"
}))
print(f"Result: {data}")

# Test 4: Try with posSide=long cross
print("\n--- Test cross with posSide=long ---")
data = client._make_request("POST", "/api/v5/trade/order", json.dumps({
    "instId": "DOGE-USDT-SWAP",
    "tdMode": "cross",
    "side": "buy",
    "ordType": "limit",
    "sz": "1",
    "px": "0.05",
    "lever": "3",
    "posSide": "long"
}))
print(f"Result: {data}")
